import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HybridResponse:
    mode: str
    output: str
    latency_ms: float
    safety_passed: bool
    reason: str


class SafetyGuard:
    """Input validation and basic safety/integrity checks."""

    BLOCKED_PATTERNS = [
        r"\b(malware|ransomware|exploit payload|sql injection payload)\b",
    ]

    @classmethod
    def validate_input(cls, prompt: str) -> tuple[bool, str]:
        if not prompt or not prompt.strip():
            return False, "Empty prompt is not allowed."
        if len(prompt) > 10000:
            return False, "Prompt exceeds maximum allowed length (10,000 chars)."
        lowered = prompt.lower()
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, lowered):
                return False, "Input blocked by safety policy."
        return True, "OK"

    @classmethod
    def verify_output(cls, text: str) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, "Empty model output."
        return True, "Verified"


class LocalEdgeEngine:
    """
    Lightweight local processing engine.
    Attempts to use local HuggingFace transformers if available;
    otherwise uses a fast, deterministic local edge processor.
    """

    def __init__(self, model_name: str = "sshleifer/tiny-gpt2") -> None:
        self.model_name = model_name
        self._hf_pipeline = None
        self._hf_checked = False

    def _get_hf_pipeline(self):
        if self._hf_checked:
            return self._hf_pipeline
        self._hf_checked = True
        try:
            from transformers import pipeline  # type: ignore

            self._hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        except Exception:
            self._hf_pipeline = None
        return self._hf_pipeline

    def run(self, prompt: str, use_hf_if_available: bool = False) -> str:
        cleaned = " ".join(prompt.strip().split())

        if use_hf_if_available:
            pipe = self._get_hf_pipeline()
            if pipe is not None:
                try:
                    result = pipe(
                        cleaned,
                        max_new_tokens=60,
                        do_sample=False,
                        truncation=True,
                    )
                    generated = result[0].get("generated_text", "").strip()
                    if generated:
                        return generated
                except Exception:
                    pass

        # Fast, real local edge analysis & structured response
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        key_terms = sorted(
            set(w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", cleaned))
        )[:6]

        summary = sentences[0] if sentences else cleaned[:160]
        terms_str = ", ".join(key_terms) if key_terms else "general-task"

        return (
            f"[Local Edge Engine] Processed locally ({word_count} words, {char_count} chars). "
            f"Key topics: [{terms_str}]. "
            f"Summary: {summary}"
        )


class CloudClient:
    """Optional cloud inference client using standard library urllib."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_sec: float = 8.0,
    ) -> None:
        self.api_url = api_url or os.getenv("HYBRIDGEN_CLOUD_URL", "").strip()
        self.api_key = api_key or os.getenv("HYBRIDGEN_CLOUD_API_KEY", "").strip()
        self.model = model or os.getenv("HYBRIDGEN_CLOUD_MODEL", "gpt-4o-mini")
        self.timeout_sec = timeout_sec

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.is_configured():
            raise RuntimeError("Cloud endpoint or API key not configured.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise, accurate AI assistant inside HybridGen AI.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                choices = parsed.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "").strip()
                raise RuntimeError("Unexpected cloud response format.")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cloud request failed: {exc}") from exc


class HybridGenRouter:
    """Dynamically routes tasks between Local Edge and Cloud with safe fallback."""

    def __init__(
        self,
        local_engine: Optional[LocalEdgeEngine] = None,
        cloud_client: Optional[CloudClient] = None,
        complexity_char_threshold: int = 220,
    ) -> None:
        self.local_engine = local_engine or LocalEdgeEngine()
        self.cloud_client = cloud_client or CloudClient()
        self.complexity_char_threshold = complexity_char_threshold

    def _should_use_cloud(self, prompt: str, force_local: bool) -> tuple[bool, str]:
        if force_local:
            return False, "Local execution forced for privacy/low-latency."
        if not self.cloud_client.is_configured():
            return False, "Cloud not configured; using local edge engine."

        heavy_keywords = [
            "architecture",
            "analyze deeply",
            "comprehensive",
            "multi-step",
            "research",
            "compare in detail",
        ]
        is_complex = len(prompt) >= self.complexity_char_threshold or any(
            kw in prompt.lower() for kw in heavy_keywords
        )
        if is_complex:
            return True, "Complex or long task routed to cloud."
        return False, "Lightweight task handled locally for low latency."

    def execute(self, prompt: str, force_local: bool = False) -> HybridResponse:
        start = time.perf_counter()

        valid, msg = SafetyGuard.validate_input(prompt)
        if not valid:
            elapsed = (time.perf_counter() - start) * 1000
            return HybridResponse(
                mode="blocked",
                output=msg,
                latency_ms=round(elapsed, 2),
                safety_passed=False,
                reason="Input failed safety validation.",
            )

        use_cloud, reason = self._should_use_cloud(prompt, force_local=force_local)

        if use_cloud:
            try:
                cloud_text = self.cloud_client.generate(prompt)
                ok, verify_msg = SafetyGuard.verify_output(cloud_text)
                elapsed = (time.perf_counter() - start) * 1000
                if ok:
                    return HybridResponse(
                        mode="cloud",
                        output=cloud_text,
                        latency_ms=round(elapsed, 2),
                        safety_passed=True,
                        reason=f"{reason} ({verify_msg})",
                    )
            except Exception as exc:
                # Seamless fallback to local engine if cloud fails or is offline
                local_text = self.local_engine.run(prompt)
                ok, verify_msg = SafetyGuard.verify_output(local_text)
                elapsed = (time.perf_counter() - start) * 1000
                return HybridResponse(
                    mode="local-fallback",
                    output=local_text,
                    latency_ms=round(elapsed, 2),
                    safety_passed=ok,
                    reason=f"Cloud unavailable ({exc}); fell back to local ({verify_msg}).",
                )

        local_text = self.local_engine.run(prompt)
        ok, verify_msg = SafetyGuard.verify_output(local_text)
        elapsed = (time.perf_counter() - start) * 1000
        return HybridResponse(
            mode="local",
            output=local_text,
            latency_ms=round(elapsed, 2),
            safety_passed=ok,
            reason=f"{reason} ({verify_msg})",
        )


if __name__ == "__main__":
    router = HybridGenRouter()

    sample_prompts = [
        ("Check sensor reading 36.8C normal.", True),
        (
            "Design a comprehensive hybrid edge-cloud architecture for real-time telemetry, "
            "offline resilience, and multi-step scientific simulation across distributed nodes.",
            False,
        ),
    ]

    for text, local_only in sample_prompts:
        res = router.execute(text, force_local=local_only)
        print(f"Mode: {res.mode} | Latency: {res.latency_ms} ms | Safe: {res.safety_passed}")
        print(f"Reason: {res.reason}")
        print(f"Output: {res.output}\n" + "-" * 60)
