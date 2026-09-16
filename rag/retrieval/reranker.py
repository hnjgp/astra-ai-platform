import re


class KeywordReranker:
    def rerank(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("query must not be empty")

        query_words = self._tokenize(query)

        scored_results = []

        for result in results:
            content_words = self._tokenize(
                result["content"]
            )

            keyword_score = len(
                query_words & content_words
            )

            scored_results.append(
                (
                    keyword_score,
                    result,
                )
            )

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in scored_results
        ]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )