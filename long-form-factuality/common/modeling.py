# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sets up language models to be used."""

from concurrent import futures
import functools
import logging
import os
import threading
import time
from typing import Any, Annotated, Optional, cast

import anthropic
import langfun as lf
import openai
import pyglove as pg
from openai import error as openai_error
from openai import openai_object
from vllm import LLM, SamplingParams

# pylint: disable=g-bad-import-order
from common import modeling_utils
from common import shared_config
from common import utils
# pylint: enable=g-bad-import-order

_DEBUG_PRINT_LOCK = threading.Lock()
_ANTHROPIC_MODELS = [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-2.1',
    'claude-2.0',
    'claude-instant-1.2',
]


class Usage(pg.Object):
  """Usage information per completion."""

  prompt_tokens: int
  completion_tokens: int


class LMSamplingResult(lf.LMSamplingResult):
  """LMSamplingResult with usage information."""

  usage: Usage | None = None


@lf.use_init_args(['model'])
class AnthropicModel(lf.LanguageModel):
  """Anthropic model."""

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, _ANTHROPIC_MODELS),
      'The name of the model to use.',
  ] = 'claude-instant-1.2'
  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'ANTHROPIC_API_KEY'."
      ),
  ] = None

  def _on_bound(self) -> None:
    super()._on_bound()
    self.__dict__.pop('_api_initialized', None)

  @functools.cached_property
  def _api_initialized(self) -> bool:
    self.api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY', None)

    if not self.api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `ANTHROPIC_API_KEY` with your Anthropic API key.'
      )

    return True

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'Anthropic({self.model})'

  def _get_request_args(
      self, options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    # Reference: https://docs.anthropic.com/claude/reference/messages_post
    args = dict(
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        stream=False,
        model=self.model,
    )

    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.top_k is not None:
      args['top_k'] = options.top_k
    if options.stop:
      args['stop_sequences'] = options.stop

    return args

  def _sample(self, prompts: list[lf.Message]) -> list[LMSamplingResult]:
    assert self._api_initialized
    return self._complete_batch(prompts)

  def _set_logging(self) -> None:
    logger: logging.Logger = logging.getLogger('anthropic')
    httpx_logger: logging.Logger = logging.getLogger('httpx')
    logger.setLevel(logging.WARNING)
    httpx_logger.setLevel(logging.WARNING)

  def _complete_batch(
      self, prompts: list[lf.Message]
  ) -> list[LMSamplingResult]:
    def _anthropic_chat_completion(prompt: lf.Message) -> LMSamplingResult:
      content = prompt.text
      client = anthropic.Anthropic(api_key=self.api_key)
      response = client.messages.create(
          messages=[{'role': 'user', 'content': content}],
          **self._get_request_args(self.sampling_options),
      )
      model_response = response.content[0].text
      samples = [lf.LMSample(model_response, score=0.0)]
      return LMSamplingResult(
          samples=samples,
          usage=Usage(
              prompt_tokens=response.usage.input_tokens,
              completion_tokens=response.usage.output_tokens,
          ),
      )

    self._set_logging()
    return lf.concurrent_execute(
        _anthropic_chat_completion,
        prompts,
        executor=self.resource_id,
        max_workers=1,
        max_attempts=self.max_attempts,
        retry_interval=self.retry_interval,
        exponential_backoff=self.exponential_backoff,
        retry_on_errors=(
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ),
    )
  

@lf.use_init_args(['model', 'base_url', 'stop_token_ids'])
class VllmModel(lf.LanguageModel):
  """vllm model."""

  model: pg.typing.Annotated[
      str,
      'The name of the model to use.',
  ] = 'google/gemma-1.1-2b-it'

  base_url: pg.typing.Annotated[
      str,
      'The url to the vllm server',
  ] = ''

  stop_token_ids: pg.typing.Annotated[
      list[int] | None,
      'The stop token ids for the model',
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('_api_initialized', None)

  @functools.cached_property
  def _api_initialized(self):
    """No API key is required for vllm. Just set the base_url."""
    openai.api_base = self.base_url
    openai.api_key = "vllm"
    return True

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'vllm({self.model})'

  @property
  def max_concurrency(self) -> int:
    """Hardcoded max concurrency for vllm."""
    return 4

  @classmethod
  def dir(cls):
    """See https://docs.vllm.ai/en/stable/models/supported_models.html for supported models."""
    return []

  @property
  def is_chat_model(self):
    return True

  def _get_request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    args = dict(
        n=options.n,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        stream=False,
        timeout=self.timeout,
    )
    args['model'] = self.model

    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    elif self.stop_token_ids:
      args['stop_token_ids'] = self.stop_token_ids
    return args

  def _sample(self, prompts: list[lf.Message]) -> list[lf.llms.openai.LMSamplingResult]:
    assert self._api_initialized
    return self._chat_complete_batch(prompts)

  def _chat_complete_batch(
      self, prompts: list[lf.Message]
  ) -> list[lf.llms.openai.LMSamplingResult]:
    def _open_ai_chat_completion(prompt: lf.Message):
      content = prompt.text

      response = openai.ChatCompletion.create(
          messages=[{'role': 'user', 'content': content}],
          **self._get_request_args(self.sampling_options),
      )
      response = cast(openai_object.OpenAIObject, response)
      return lf.llms.openai.LMSamplingResult(
          [
              lf.LMSample(choice.message.content, score=0.0)
              for choice in response.choices
          ],
          usage=lf.llms.openai.Usage(
              prompt_tokens=response.usage.prompt_tokens,
              completion_tokens=response.usage.completion_tokens,
              total_tokens=response.usage.total_tokens,
          ),
      )

    return lf.concurrent_execute(
        _open_ai_chat_completion,
        prompts,
        executor=self.resource_id,
        max_workers=self.max_concurrency,
        retry_on_errors=(
            openai_error.ServiceUnavailableError,
            openai_error.RateLimitError,
        ),
        max_attempts=self.max_attempts,
        retry_interval=self.retry_interval,
        exponential_backoff=self.exponential_backoff,
    )


@lf.use_init_args(['model', 'stop_token_ids'])
class VllmOfflineModel(lf.LanguageModel):
  """vllm offline model."""

  model: pg.typing.Annotated[
      Any,
      'The vllm LLM model',
  ] = None

  stop_token_ids: pg.typing.Annotated[
      list[int] | None,
      'The stop token ids for the model',
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('_api_initialized', None)

  @functools.cached_property
  def _api_initialized(self):
    return True

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'vllm({self.model})'

  @property
  def max_concurrency(self) -> int:
    """Hardcoded max concurrency for vllm."""
    return 1

  @classmethod
  def dir(cls):
    """See https://docs.vllm.ai/en/stable/models/supported_models.html for supported models."""
    return []

  @property
  def is_chat_model(self):
    return True

  def _get_request_args(
      self, options: lf.LMSamplingOptions) -> SamplingParams:
    args = dict(
        n=options.n,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )

    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    elif self.stop_token_ids:
      args['stop_token_ids'] = self.stop_token_ids
    return SamplingParams(**args)

  def _sample(self, prompts: list[lf.Message]) -> list[lf.llms.openai.LMSamplingResult]:
    assert self._api_initialized
    return self._chat_complete_batch(prompts)

  def _chat_complete_batch(
      self, prompts: list[lf.Message]
  ) -> list[lf.llms.openai.LMSamplingResult]:
    prompts = [prompt.text for prompt in prompts]
    sampling_params = self._get_request_args(self.sampling_options)
    
    outputs = self.model.generate(prompts, sampling_params)
    results = []
    for output in outputs:
      result = lf.llms.openai.LMSamplingResult(
        [
            lf.LMSample(choice.text, score=0.0)
            for choice in output.outputs
        ],
        usage=lf.llms.openai.Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
      )
      results.append(result)

    return results


class Model:
  """Class for storing any single language model."""

  def __init__(
      self,
      model_name: str,
      temperature: float = 0.5,
      max_tokens: int = 2048,
      show_responses: bool = False,
      show_prompts: bool = False,
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.show_responses = show_responses
    self.show_prompts = show_prompts
    self.model = self.load(model_name, self.temperature, self.max_tokens)

  def load(
      self, model_name: str, temperature: float, max_tokens: int
  ) -> lf.LanguageModel:
    """Loads a language model from string representation."""
    sampling = lf.LMSamplingOptions(
        temperature=temperature, max_tokens=max_tokens
    )

    if model_name.lower().startswith('openai:'):
      if not shared_config.openai_api_key:
        utils.maybe_print_error('No OpenAI API Key specified.')
        utils.stop_all_execution(True)

      return lf.llms.OpenAI(
          model=model_name[7:],
          api_key=shared_config.openai_api_key,
          sampling_options=sampling,
      )
    elif model_name.lower().startswith('anthropic:'):
      if not shared_config.anthropic_api_key:
        utils.maybe_print_error('No Anthropic API Key specified.')
        utils.stop_all_execution(True)

      return AnthropicModel(
          model=model_name[10:],
          api_key=shared_config.anthropic_api_key,
          sampling_options=sampling,
      )
    elif model_name.lower().startswith('vllm:'):
      if not shared_config.vllm_server_url:
        utils.maybe_print_error('No vllm server url specified.')
        utils.stop_all_execution(True)

      return VllmModel(
          model=model_name[5:],
          base_url=shared_config.vllm_server_url,
          stop_token_ids=shared_config.vllm_stop_token_ids,
          sampling_options=sampling,
      )
    elif model_name.lower().startswith('vllm_offline:'):
      llm = LLM(model_name[13:], enforce_eager=True)
      return VllmOfflineModel(
          model=llm,
          stop_token_ids=shared_config.vllm_stop_token_ids,
          sampling_options=sampling,
      )
    elif 'unittest' == model_name.lower():
      return lf.llms.Echo()
    else:
      raise ValueError(f'ERROR: Unsupported model type: {model_name}.')

  def generate(
      self,
      prompt: str,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 5,
      timeout: int = 60,
      retry_interval: int = 10,
  ) -> str:
    """Generates a response to a prompt."""
    self.model.max_attempts = 1
    self.model.retry_interval = 0
    self.model.timeout = timeout
    prompt = modeling_utils.add_format(prompt, self.model, self.model_name)
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens
    response, num_attempts = '', 0

    with modeling_utils.get_lf_context(gen_temp, gen_max_tokens):
      while not response and num_attempts < max_attempts:
        with futures.ThreadPoolExecutor() as executor:
          future = executor.submit(lf.LangFunc(prompt, lm=self.model))

          try:
            response = future.result(timeout=timeout).text
          except (
              openai.error.OpenAIError,
              futures.TimeoutError,
              lf.core.concurrent.RetryError,
              anthropic.AnthropicError,
          ) as e:
            utils.maybe_print_error(e)
            time.sleep(retry_interval)

        num_attempts += 1

    # import json
    # with _DEBUG_PRINT_LOCK:
    #   with open('/work/cwhuang0921/long-form-factuality/debug.jsonl', 'a') as f:
    #     f.write(json.dumps({'prompt': prompt, 'response': response}) + "\n")

    if do_debug:
      with _DEBUG_PRINT_LOCK:
        if self.show_prompts:
          utils.print_color(prompt, 'magenta')
        if self.show_responses:
          utils.print_color(response, 'cyan')

    return response

  def print_config(self) -> None:
    settings = {
        'model_name': self.model_name,
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'show_responses': self.show_responses,
        'show_prompts': self.show_prompts,
    }
    print(utils.to_readable_json(settings))


class FakeModel(Model):
  """Class for faking responses during unit tests."""

  def __init__(
      self,
      static_response: str = '',
      sequential_responses: Optional[list[str]] = None,
  ) -> None:
    Model.__init__(self, model_name='unittest')
    self.static_response = static_response
    self.sequential_responses = sequential_responses
    self.sequential_response_idx = 0

    if static_response:
      self.model = lf.llms.StaticResponse(static_response)
    elif sequential_responses:
      self.model = lf.llms.StaticSequence(sequential_responses)
    else:
      self.model = lf.llms.Echo()

  def generate(
      self,
      prompt: str,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 1000,
      timeout: int = 60,
      retry_interval: int = 10,
  ) -> str:
    if self.static_response:
      return self.static_response
    elif self.sequential_responses:
      response = self.sequential_responses[
          self.sequential_response_idx % len(self.sequential_responses)
      ]
      self.sequential_response_idx += 1
      return response
    else:
      return ''
