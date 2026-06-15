import os
import time
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RateLimiter:
    def __init__(self, keys, max_rpm=4, max_tpm=240000, max_rpd=19):
        """
        Manages state for multiple API keys tracking RPM, TPM, and RPD.
        """
        self.keys = keys
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.max_rpd = max_rpd
        
        self.idx = 0
        
        # Track limits per key
        self.rpd_counts = {k: 0 for k in keys}
        self.rpm_counts = {k: 0 for k in keys}
        self.tpm_counts = {k: 0 for k in keys}
        
        self.minute_start = time.time()
        self.day_start = time.time()
        
    def _reset_time_windows(self, now):
        # Reset minute window
        if now - self.minute_start >= 60:
            for k in self.keys:
                self.rpm_counts[k] = 0
                self.tpm_counts[k] = 0
            self.minute_start = now
            
        # Reset day window
        if now - self.day_start >= 86400:
            for k in self.keys:
                self.rpd_counts[k] = 0
            self.day_start = now

    def check_limits_and_rotate(self, estimated_tokens=500):
        now = time.time()
        self._reset_time_windows(now)
        
        current_key = self.keys[self.idx]
        
        # Check RPD limit
        if self.rpd_counts[current_key] >= self.max_rpd:
            print(f"[API Manager] Key {self.idx + 1} reached RPD limit ({self.max_rpd}). Switching key...")
            self.idx = (self.idx + 1) % len(self.keys)
            current_key = self.keys[self.idx]
        
        # Check RPM and TPM limits
        if self.rpm_counts[current_key] >= self.max_rpm or self.tpm_counts[current_key] + estimated_tokens >= self.max_tpm:
            sleep_time = 60 - (now - self.minute_start) + 1
            if sleep_time > 0:
                print(f"[API Manager] Key {self.idx + 1} reached RPM/TPM limit. Pausing for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                # After sleeping, time window will be reset on next call
                return self.check_limits_and_rotate(estimated_tokens)
                
        # Register usage
        self.rpm_counts[current_key] += 1
        self.rpd_counts[current_key] += 1
        self.tpm_counts[current_key] += estimated_tokens

    async def acheck_limits_and_rotate(self, estimated_tokens=500):
        now = time.time()
        self._reset_time_windows(now)
        
        current_key = self.keys[self.idx]
        
        if self.rpd_counts[current_key] >= self.max_rpd:
            print(f"[API Manager] Key {self.idx + 1} reached RPD limit ({self.max_rpd}). Switching key...")
            self.idx = (self.idx + 1) % len(self.keys)
            current_key = self.keys[self.idx]
            
        if self.rpm_counts[current_key] >= self.max_rpm or self.tpm_counts[current_key] + estimated_tokens >= self.max_tpm:
            sleep_time = 60 - (now - self.minute_start) + 1
            if sleep_time > 0:
                print(f"[API Manager] Key {self.idx + 1} reached RPM/TPM limit. Pausing for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                return await self.acheck_limits_and_rotate(estimated_tokens)
                
        self.rpm_counts[current_key] += 1
        self.rpd_counts[current_key] += 1
        self.tpm_counts[current_key] += estimated_tokens


def _create_rotating_ragas_llm(llms, rate_limiter):
    # Ragas >= 0.2 moved LangchainLLMWrapper. Import from actual module, not the deprecated alias.
    try:
        from ragas.llms.base import LangchainLLMWrapper
    except ImportError:
        from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    
    class RotatingRagasLLM(LangchainLLMWrapper):
        def __init__(self, llms, rate_limiter):
            self.llms = llms
            self.rate_limiter = rate_limiter
            super().__init__(run_config=RunConfig(), langchain_llm=llms[0])

        def generate(self, prompts, n=1, temperature=0.0, callbacks=None):
            from google.genai.errors import ClientError as GoogleClientError
            max_attempts = len(self.llms)
            for attempt in range(max_attempts):
                self.rate_limiter.check_limits_and_rotate()
                current_key_str = self.rate_limiter.keys[self.rate_limiter.idx]
                self.langchain_llm = self.llms[self.rate_limiter.idx]
                try:
                    return super().generate(prompts, n, temperature, callbacks)
                except GoogleClientError as e:
                    if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                        print(f"\n[API WARNING RAGAS] Key {self.rate_limiter.idx + 1} exhausted! Rotating...")
                        self.rate_limiter.rpd_counts[current_key_str] = 9999
                        continue
                    raise e
                except Exception as e:
                    if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                        print(f"\n[API WARNING RAGAS] Key {self.rate_limiter.idx + 1} exhausted! Rotating...")
                        self.rate_limiter.rpd_counts[current_key_str] = 9999
                        continue
                    raise e
            raise RuntimeError("Semua API Keys telah habis kuotanya (RESOURCE_EXHAUSTED) saat evaluasi Ragas.")

        async def agenerate(self, prompt, n=1, temperature=0.0, callbacks=None):
            from google.genai.errors import ClientError as GoogleClientError
            max_attempts = len(self.llms)
            for attempt in range(max_attempts):
                await self.rate_limiter.acheck_limits_and_rotate()
                current_key_str = self.rate_limiter.keys[self.rate_limiter.idx]
                self.langchain_llm = self.llms[self.rate_limiter.idx]
                try:
                    return await super().agenerate(prompt, n, temperature, callbacks)
                except GoogleClientError as e:
                    if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                        print(f"\n[API WARNING RAGAS] Key {self.rate_limiter.idx + 1} exhausted! Rotating...")
                        self.rate_limiter.rpd_counts[current_key_str] = 9999
                        continue
                    raise e
                except Exception as e:
                    if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                        print(f"\n[API WARNING RAGAS] Key {self.rate_limiter.idx + 1} exhausted! Rotating...")
                        self.rate_limiter.rpd_counts[current_key_str] = 9999
                        continue
                    raise e
            raise RuntimeError("Semua API Keys telah habis kuotanya (RESOURCE_EXHAUSTED) saat evaluasi Ragas.")
            
    return RotatingRagasLLM(llms, rate_limiter)

class RotatingChatWrapper:
    """
    Simple wrapper for direct LLM invoke() calls in sandbox / regular inference.
    Not for Ragas internal generation, but acts like a BaseChatModel for invoke.
    """
    def __init__(self, llms, rate_limiter: RateLimiter):
        self.llms = llms
        self.rate_limiter = rate_limiter
        
    def invoke(self, prompt, **kwargs):
        from google.genai.errors import ClientError as GoogleClientError
        
        max_attempts = len(self.llms)
        for attempt in range(max_attempts):
            self.rate_limiter.check_limits_and_rotate()
            current_key_str = self.rate_limiter.keys[self.rate_limiter.idx]
            try:
                return self.llms[self.rate_limiter.idx].invoke(prompt, **kwargs)
            except GoogleClientError as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    print(f"\n[API WARNING] Key {self.rate_limiter.idx + 1} exhausted on server side! Rotating...")
                    self.rate_limiter.rpd_counts[current_key_str] = 9999  # Force rotate
                    continue
                raise e
            except Exception as e:
                # Catch wrapped langchain exceptions
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    print(f"\n[API WARNING] Key {self.rate_limiter.idx + 1} exhausted on server side! Rotating...")
                    self.rate_limiter.rpd_counts[current_key_str] = 9999  # Force rotate
                    continue
                raise e
        
        raise RuntimeError("Semua API Keys telah habis kuotanya (RESOURCE_EXHAUSTED).")

def get_api_manager(eval_mode: bool = False):
    """
    Helper to instantiate API keys from .env and return the limiter and LLMs.
    
    Args:
        eval_mode: If True, also initializes the Ragas evaluator_llm wrapper.
                   Set to False (default) for Sandbox usage to avoid ragas import errors.
    """
    keys = []
    for i in range(1, 4):
        k = os.environ.get(f"GEMINI_KEY_{i}")
        if k:
            keys.append(k)
            
    if not keys:
        # Fallback to general env variable if GEMINI_KEY_1..3 are not set
        k = os.environ.get("GEMINI_API_KEY")
        if k:
            keys.append(k)
            
    if not keys:
        raise ValueError("No Gemini API keys found. Please set GEMINI_KEY_1, 2, 3 in .env file.")
        
    # Instantiate Chat models
    llms_inference = [ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_retries=0, google_api_key=k) for k in keys]
    llms_eval = [ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=0, google_api_key=k) for k in keys]
    
    rate_limiter = RateLimiter(keys)
    
    # Ragas Wrappers: only load when explicitly requested (eval_mode=True)
    evaluator_llm = None
    if eval_mode:
        try:
            evaluator_llm = _create_rotating_ragas_llm(llms_eval, rate_limiter)
        except (ImportError, TypeError) as e:
            print(f"[API Manager] Warning: Could not init Ragas LLM wrapper: {e}")
            evaluator_llm = None
        
    # Langchain Wrapper for Sandbox & Judge
    invoke_llm = RotatingChatWrapper(llms_inference, rate_limiter)
    judge_llm = RotatingChatWrapper(llms_eval, rate_limiter)
    
    return invoke_llm, evaluator_llm, judge_llm
