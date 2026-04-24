import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    request_id: int
    session_id: int
    turn: int
    full_prompt: str
    prefix_prompt: str
    new_prompt: str
    prefix_hash: str
    is_first_turn: bool

@dataclass
class WorkloadStats:
    total_requests: int
    unique_prefixes: int
    expected_hit_rate: float
    avg_prefix_tokens: float
    avg_new_tokens: float

class ShareGPTWorkload:

    SHORT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant. Please answer the user's questions "
        "accurately and concisely. Provide detailed explanations when asked."
    )

    LONG_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert AI assistant deployed in a large-scale enterprise environment. "
        "Your role is to provide accurate, detailed, and well-structured answers based on "
        "the following reference documentation and guidelines.\n\n"
        "=== REFERENCE DOCUMENTATION ===\n"
        "{padding}\n"
        "=== END OF REFERENCE ===\n\n"
        "Based on the above documentation, answer the user's questions accurately. "
        "Always cite specific sections when possible. Provide step-by-step explanations."
    )

    SYSTEM_PROMPT = SHORT_SYSTEM_PROMPT

    def __init__(self, dataset_path: str, tokenizer=None):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.conversations = []

    def set_prefix_length(self, target_tokens: int, tokenizer=None):
        tok = tokenizer or self.tokenizer
        if target_tokens <= 100:
            self.SYSTEM_PROMPT = self.SHORT_SYSTEM_PROMPT
            return

        padding_sentences = [
            "Section {i}: Distributed systems require careful consideration of network partitions, "
            "fault tolerance mechanisms, consistency models, and replication strategies. "
            "The CAP theorem states that a distributed system cannot simultaneously provide "
            "consistency, availability, and partition tolerance. Modern systems must carefully "
            "balance these trade-offs based on application requirements and workload patterns. "
            "Key considerations include leader election protocols, consensus algorithms such as "
            "Paxos and Raft, vector clocks for causal ordering, and gossip protocols for "
            "eventual consistency. Performance optimization techniques include batching, "
            "pipelining, and speculative execution to hide network latency.\n\n"
        ]

        padding = ""
        i = 0
        while True:
            padding += padding_sentences[0].format(i=i)
            i += 1
            test_prompt = self.LONG_SYSTEM_PROMPT_TEMPLATE.format(padding=padding)
            if tok:
                n_tokens = len(tok(test_prompt)["input_ids"])
                if n_tokens >= target_tokens:
                    break
            else:
                if len(test_prompt.split()) >= target_tokens * 0.75:
                    break
            if i > 500:
                break

        self.SYSTEM_PROMPT = self.LONG_SYSTEM_PROMPT_TEMPLATE.format(padding=padding)
        if tok:
            actual = len(tok(self.SYSTEM_PROMPT)["input_ids"])
            print(f"System prompt set to {actual} tokens (target: {target_tokens})")

    def load(self, max_conversations: int = 200) -> int:
        if self.dataset_path.exists():
            with open(self.dataset_path) as f:
                raw = json.load(f)

            for conv in raw[:max_conversations]:
                turns = conv.get("conversations", conv.get("items", []))
                if len(turns) >= 2:
                    self.conversations.append(turns)

            print(f"Loaded {len(self.conversations)} conversations from {self.dataset_path}")
        else:
            print(f"ShareGPT not found at {self.dataset_path}, using synthetic conversations")
            self._generate_synthetic(max_conversations)

        return len(self.conversations)

    def _generate_synthetic(self, num_conversations: int):
        topics = [
            "distributed systems", "machine learning", "climate science",
            "quantum computing", "database optimization", "network protocols",
            "operating systems", "compiler design", "cryptography",
            "parallel computing", "cloud architecture", "data structures",
        ]

        for i in range(num_conversations):
            topic = topics[i % len(topics)]
            turns = [
                {"from": "human", "value": f"Explain the fundamentals of {topic} and why it matters in modern computing."},
                {"from": "gpt", "value": f"{topic.title()} is a critical area..."},
                {"from": "human", "value": f"Can you give me a specific example of {topic} in practice?"},
                {"from": "gpt", "value": f"A practical example of {topic}..."},
                {"from": "human", "value": f"What are the main challenges and open problems in {topic}?"},
                {"from": "gpt", "value": f"The key challenges in {topic} include..."},
            ]
            self.conversations.append(turns)

    def generate_requests(
        self,
        num_sessions: int = 100,
        turns_per_session: int = 3,
    ) -> List[InferenceRequest]:
        requests = []
        request_id = 0

        for session_id in range(num_sessions):
            conv = self.conversations[session_id % len(self.conversations)]

            history = self.SYSTEM_PROMPT + "\n\n"
            max_turns = min(turns_per_session, len(conv) // 2)

            for turn in range(max_turns):
                human_idx = turn * 2
                if human_idx >= len(conv):
                    break

                human_msg = conv[human_idx].get("value", "")
                gpt_msg = conv[human_idx + 1].get("value", "") if human_idx + 1 < len(conv) else ""

                prefix_prompt = history
                new_prompt = f"Human: {human_msg}\nAssistant:"
                full_prompt = prefix_prompt + new_prompt

                prefix_hash = hashlib.sha256(prefix_prompt.encode()).hexdigest()[:32]

                requests.append(InferenceRequest(
                    request_id=request_id,
                    session_id=session_id,
                    turn=turn,
                    full_prompt=full_prompt,
                    prefix_prompt=prefix_prompt,
                    new_prompt=new_prompt,
                    prefix_hash=prefix_hash,
                    is_first_turn=(turn == 0),
                ))
                request_id += 1

                history += f"Human: {human_msg}\nAssistant: {gpt_msg}\n\n"

        unique_prefixes = len(set(r.prefix_hash for r in requests))
        first_turns = sum(1 for r in requests if r.is_first_turn)

        stats = WorkloadStats(
            total_requests=len(requests),
            unique_prefixes=unique_prefixes,
            expected_hit_rate=1.0 - (unique_prefixes / len(requests)) if requests else 0,
            avg_prefix_tokens=0,
            avg_new_tokens=0,
        )

        print(f"Generated {stats.total_requests} requests "
              f"({stats.unique_prefixes} unique prefixes, "
              f"expected hit rate: {stats.expected_hit_rate:.1%})")

        return requests

    def get_stats(self, requests: List[InferenceRequest]) -> WorkloadStats:
        unique_prefixes = len(set(r.prefix_hash for r in requests))
        return WorkloadStats(
            total_requests=len(requests),
            unique_prefixes=unique_prefixes,
            expected_hit_rate=1.0 - (unique_prefixes / len(requests)) if requests else 0,
            avg_prefix_tokens=0,
            avg_new_tokens=0,
        )
