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
"""Class for querying the ColBERT retrieval server."""

import random
import time
from typing import Any

import requests

NO_RESULT_MSG = 'No good search result was found'


class ColBERTAPI:
  """Class for querying the ColBERT retrieval server."""

  def __init__(
      self,
      server_url: str,
      k: int = 1
  ):
    self.server_url = server_url
    self.k = k

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through ColBERT retrieval server and parse result."""
    results = self._colbert_api_results(
        query,
        **kwargs,
    )

    return self._parse_results(results)

  def _colbert_api_results(
      self,
      search_term: str,
      max_retries: int = 5,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through ColBERT server."""
    params = {
        'query': search_term,
        'k': self.k,
    }
    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        # GET request to the server
        response = requests.get(self.server_url, params=params, **kwargs)
      except AssertionError as e:
        raise e
      except Exception:  # pylint: disable=broad-exception-caught
        response = None
        num_fails += 1
        sleep_time = min(sleep_time * 2, 60)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from ColBERT server API')

    search_results = response.json()
    return search_results

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results."""
    snippets = []
    for result in results['topk'][:self.k]:
      if 'text' in result:
        snippets.append(result['text'])

    if not snippets:
      return [NO_RESULT_MSG]

    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' | '.join(self._parse_snippets(results))
