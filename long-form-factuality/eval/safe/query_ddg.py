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
"""Class for querying the DuckDuckGo API."""

import random
import time
from typing import Any, Optional, Literal

from duckduckgo_search import DDGS


NO_RESULT_MSG = 'No good search result was found'


class DDGAPI:
  """Class for querying the DuckDuckGo API."""

  def __init__(
      self,
      gl: str = 'us',
      hl: str = 'en',
      k: int = 1,
      tbs: Optional[str] = None,
      search_type: Literal['news', 'search', 'places', 'images'] = 'search',
  ):
    self.reset_ddg()
    self.gl = gl
    self.hl = hl
    self.region = f"{gl}-{hl}"
    self.k = k
    self.tbs = tbs
    self.search_type = search_type
    self.result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }
  
  def reset_ddg(self):
    self.ddg = DDGS()

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through DuckDuckGoSearch and parse result."""
    results = self._ddg_api_results(
        query,
        gl=self.gl,
        hl=self.hl,
        num=self.k,
        tbs=self.tbs,
        search_type=self.search_type,
        **kwargs,
    )

    return self._parse_results(results)

  def _ddg_api_results(
      self,
      search_term: str,
      search_type: str = 'search',
      max_retries: int = 20,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through DuckDuckGo."""
    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        if search_type == 'search':
          response = self.ddg.text(
            keywords=search_term,
            region=self.region,
            safesearch='off',
            max_results=self.k
          )
        elif search_type == 'news':
          response = self.ddg.news(
            keywords=search_term,
            region=self.region,
            safesearch='off',
            max_results=self.k
          )
        elif search_type == 'images':
          response = self.ddg.images(
            keywords=search_term,
            region=self.region,
            safesearch='off',
            max_results=self.k
          )
        else:
          raise ValueError(f'Unsupported search type: {search_type}')

      except AssertionError as e:
        raise e
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'Failed to get result from DuckDuckGo API. Retrying...')
        print(f"Exception: {e}")
        response = None
        self.reset_ddg()
        num_fails += 1
        sleep_time = min(sleep_time * 2, 600)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from DuckDuckGo API')

    return response

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results."""
    snippets = []
    for result in results[:self.k]:
      if 'text' in result:
        # DuckDuckGo API returns text for answers
        snippets.append(result['text'])
      elif 'title' in result:
        # DuckDuckGo API returns title and body for text and news
        snippets.append(f"{result['title']}: {result['body']}")

    if not snippets:
      return [NO_RESULT_MSG]

    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' '.join(self._parse_snippets(results))
