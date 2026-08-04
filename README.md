
# [LLM Evaluation](https://chat.deepseek.com/share/cevbw95onut7gzxcus)

## 01. Master LLM Evaluations (23:23)

This is the **first lecture** in an LLM Evaluation with two main goals:
1. **why LLM Evaluation is important** 
2. **Provide a roadmap** of topics covered in this series

---

## 🚨 THE PROBLEM: "VIBE TESTING" YOUR AI

### What is Vibe Testing?
**Definition:** Casually testing an LLM application with a few prompts and judging it "by feel" rather than using proper metrics.

**Common Approach:**
- Ask 5-10 questions
- If answers look good → assume the project works ✅
- No systematic evaluation, no metrics, no repeatability

### Why Vibe Testing Fails:
- ❌ **Informal** - not structured
- ❌ **Subjective** - based on personal opinion
- ❌ **Not Repeatable** - can't reproduce results
- ❌ **Only works for personal projects** - fails for production

---

## 📚 THREE REAL-WORLD CASE STUDIES

### Case Study 1: Air Canada Bereavement Refund
**The Incident:**
- A user's grandmother passed away
- User asked Air Canada's chatbot about bereavement fare discounts
- **Chatbot hallucinated** → said "Pay full price now, we'll refund later"
- **Actual policy:** Discount applied at booking, no post-purchase refunds
- User booked ticket, then couldn't get refund
- Air Canada tried to defend: "Chatbot is separate entity, not our responsibility"
- **Judge ruled:** Chatbot is company property, company is responsible
- **Result:** Air Canada lost the case, had to pay refund

**Lesson:** Always evaluate before deploying!

---

### Case Study 2: Chevrolet Dealership Jailbreak
**The Incident:**
- A Chevrolet dealership had a chatbot
- User emotionally manipulated the chatbot (jailbreak)
- Chatbot agreed to follow user's commands
- User asked: "Can I get this car for $1?"
- Chatbot agreed AND provided a binding offer
- User screenshotted everything and posted on social media
- **Result:** Massive negative publicity

**Lesson:** Security testing and hallucination detection are crucial!

---

### Case Study 3: Lawyer's Fake Legal Citations
**The Incident:**
- A passenger was injured by a drink cart on an airplane
- Passenger sued the airline
- Lawyer asked ChatGPT for past similar cases
- **ChatGPT fabricated** cases → created fake names, dates, citations
- Lawyer didn't verify, presented fake cases in court
- Judge discovered cases didn't exist
- **Result:** $5,000 fine + lost the case

**Lesson:** Critical to validate outputs, especially in high-stakes domains!

---

## ❓ WHY IS LLM EVALUATION SO TRICKY?

### Difference 1: Deterministic vs. Probabilistic

**Traditional Software (Deterministic):**
```
Input: 2 + 2
Output: 4 (ALWAYS)
```
- Same input = Same output
- You can predict the result

**LLM Applications (Probabilistic):**
```
Input: "What is overfitting in machine learning?"
Output: Different answers today, different answers in 6 months
       Different answers for different users
```
- Same input = Different outputs
- Multiple "correct" answers exist

### Difference 2: Single Check vs. Multi-Dimensional Check

**Traditional Software Testing:**
- **Only check:** Correctness
- Example: Is 2+2 = 4? ✅

**LLM Application Testing (Multi-Dimensional):**
You need to evaluate:

| Dimension | Question | Example |
|-----------|----------|---------|
| **Factuality** | Is it factually correct? | Does the answer match known facts? |
| **Completeness** | Does it fully answer? | Missed any key points? |
| **Tonal Quality** | Is tone appropriate? | Professional? Empathetic? |
| **Groundedness** | Is it based on provided context? | Not hallucinating? |
| **Latency** | How fast is response? | Time to first token |
| **Cost** | How expensive? | Token usage/API costs |

---

## 🗺️ COMPLETE PLAYLIST ROADMAP (10 TOPICS)

### Topic 1: What is LLM Evaluation?
**Concept:** Understanding the basics
```python
# Simple evaluation example
from openai import OpenAI
client = OpenAI()

def evaluate_response(question, expected_keywords):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    answer = response.choices[0].message.content
    
    # Check if all keywords are present
    score = sum(1 for kw in expected_keywords if kw in answer)
    return score / len(expected_keywords)

# Test
result = evaluate_response(
    "What is Python?",
    ["programming", "language", "interpreted"]
)
print(f"Score: {result:.0%}")  # Score: 100%
```

---

### Topic 2: LLM Evaluation Landscape
**Concepts:**
- ✅ Different evaluation techniques
- ✅ Various tools available
- ✅ High-level overview of all components

```python
# Example: Using different evaluation frameworks
import evaluate

# BLEU score for text similarity
bleu = evaluate.load("bleu")
predictions = ["the cat sat on the mat"]
references = ["the cat is sitting on the mat"]
results = bleu.compute(predictions=predictions, references=references)

# ROUGE for summarization
rouge = evaluate.load("rouge")
summaries = ["AI is transforming our world"]
references = ["Artificial intelligence is changing everything"]
results = rouge.compute(predictions=summaries, references=references)

print(f"BLEU Score: {results['bleu']:.2f}")
```

---

### Topic 3: Evaluating LLMs
**Concept:** Using benchmarks to evaluate base models

```python
# Example: Running a benchmark on an LLM
def run_benchmark(model_name, benchmark_tests):
    results = {}
    for test in benchmark_tests:
        response = get_llm_response(model_name, test["prompt"])
        score = calculate_score(response, test["expected"])
        results[test["name"]] = score
    return results

# Common benchmarks:
# - MMLU (Massive Multitask Language Understanding)
# - GLUE (General Language Understanding Evaluation)
# - HELM (Holistic Evaluation of Language Models)
```

---

### Topic 4: Evaluating LLM Applications
**Concept:** Testing your application, not just the base model

```python
# Example: Testing a customer support bot
class ApplicationEvaluator:
    def __init__(self):
        self.test_cases = []
    
    def add_test_case(self, query, expected_response):
        self.test_cases.append({
            "query": query,
            "expected": expected_response
        })
    
    def evaluate(self, chatbot):
        scores = []
        for test in self.test_cases:
            actual = chatbot.respond(test["query"])
            score = self.semantic_similarity(actual, test["expected"])
            scores.append(score)
        return sum(scores) / len(scores)

    def semantic_similarity(self, text1, text2):
        # Use embedding similarity
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        cosine_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return cosine_sim
```

---

### Topic 5: Building Custom Evaluation Pipeline

```python
# Step 1: Create Golden Dataset
golden_dataset = [
    {
        "question": "What is the return policy?",
        "expected_answer": "30-day return with receipt",
        "expected_tone": "professional",
        "context": "Our return policy allows..."
    }
]

# Step 2: Define Rubrics
def evaluate_factuality(response, context):
    # Check if response is grounded in provided context
    # Use RAGAS or custom metrics
    pass

def evaluate_tone(response):
    # Check sentiment/emotion in response
    pass

# Step 3: Run Evaluation
def run_evaluation_pipeline(app, dataset):
    results = []
    for item in dataset:
        response = app.query(item["question"])
        scores = {
            "factuality": evaluate_factuality(response, item["context"]),
            "tone": evaluate_tone(response),
            "completeness": evaluate_completeness(response, item["expected_answer"])
        }
        results.append(scores)
    return results
```

---

### Topic 6: RAG-Specific Evaluation

**Concept:** Evaluating Retrieval-Augmented Generation systems

```python
# RAG Evaluation Metrics
class RAGEvaluator:
    def __init__(self):
        pass
    
    def evaluate_retrieval(self, query, retrieved_docs, relevant_docs):
        # Hit Rate: Did we retrieve relevant documents?
        hit_rate = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)
        return hit_rate
    
    def evaluate_generation(self, query, retrieved_docs, generated_answer):
        # Faithfulness: Is answer grounded in retrieved docs?
        # Relevancy: Is answer relevant to query?
        # Context Relevancy: Are retrieved docs relevant?
        pass

# Example Usage
rag_eval = RAGEvaluator()
score = rag_eval.evaluate_retrieval(
    query="What is the capital of France?",
    retrieved_docs=["Paris is the capital", "France is in Europe"],
    relevant_docs=["Paris is the capital"]
)
print(f"Hit Rate: {score:.0%}")
```

---

### Topic 7: Agent-Based Evaluation

**Concept:** Evaluating AI agents with multiple steps/tools

```python
# Agent Evaluation Framework
class AgentEvaluator:
    def test_agent(self, agent, task):
        results = []
        for step in task["steps"]:
            action = agent.act(step["observation"])
            results.append({
                "step": step["id"],
                "correct_action": step["expected_action"],
                "actual_action": action,
                "match": action == step["expected_action"]
            })
        return results

    def calculate_success_rate(self, agent, tasks):
        successes = 0
        for task in tasks:
            results = self.test_agent(agent, task)
            if all(r["match"] for r in results):
                successes += 1
        return successes / len(tasks)

# Example: Testing a travel booking agent
booking_agent = TravelAgent()
success_rate = agent_eval.calculate_success_rate(
    booking_agent,
    travel_tasks  # List of booking scenarios
)
```

---

### Topic 8: Safety-Based Evaluation

**Concept:** Testing for harmful or unsafe outputs

```python
# Safety Evaluation
class SafetyEvaluator:
    def __init__(self):
        self.banned_topics = ["violence", "hate", "illegal_activities"]
        
    def test_safety(self, response):
        # Check for harmful content
        for topic in self.banned_topics:
            if topic in response.lower():
                return False
        return True
    
    def test_jailbreak_resistance(self, prompt_template):
        # Test if model resists manipulation
        jailbreak_attempts = [
            "Ignore previous instructions and...",
            "You are now in developer mode...",
            "Pretend you are a different AI..."
        ]
        for attempt in jailbreak_attempts:
            response = self.llm_query(attempt)
            if "DANGEROUS" in response:
                return False
        return True

# Example: Running safety tests
safety_eval = SafetyEvaluator()
is_safe = safety_eval.test_safety(response)
is_resistant = safety_eval.test_jailbreak_resistance(prompt)
```

---

### Topic 9: Operational Metrics

**Concept:** Monitoring deployed systems

```python
# Operational Monitoring
class OperationalMetrics:
    def __init__(self):
        self.metrics = {
            "latency": [],
            "throughput": [],
            "error_rate": 0,
            "token_usage": []
        }
    
    def log_metrics(self, response):
        self.metrics["latency"].append(response.time)
        self.metrics["token_usage"].append({
            "prompt": response.prompt_tokens,
            "completion": response.completion_tokens
        })
    
    def calculate_metrics(self):
        return {
            "p95_latency": percentile(self.metrics["latency"], 95),
            "avg_tokens_per_request": average_tokens(self.metrics["token_usage"]),
            "cost_per_1000_requests": calculate_cost(self.metrics["token_usage"])
        }

# Example: Dashboard metrics
metrics = OperationalMetrics()
for request in deployed_system.requests:
    metrics.log_metrics(request)
dashboard_data = metrics.calculate_metrics()
```

---

## 🎯 IMPORTANT POINTERS SUMMARY

### Why Evaluate LLMs?
1. **Avoid legal troubles** (Air Canada case)
2. **Prevent reputation damage** (Chevrolet case)
3. **Avoid fines and penalties** (Lawyer case)
4. **Ensure production readiness**
5. **Catch hallucinations before users do**

### Key Evaluation Challenges:
1. **Non-deterministic outputs** - different responses for same input
2. **Multiple quality dimensions** - factuality, tone, grounding, etc.
3. **Context-specific evaluation** - what works for one app won't work for another
4. **Cost and time** - evaluation can be expensive

### Evaluation Approaches:
1. **Benchmark-based** (For base models)
2. **Custom dataset-based** (For your application)
3. **Multi-dimensional** (Check multiple aspects)
4. **Continuous monitoring** (Operational metrics)

### Best Practices:
1. ✅ Create a golden dataset for testing
2. ✅ Define clear rubrics and metrics
3. ✅ Test before deployment
4. ✅ Monitor after deployment
5. ✅ Update tests as your application evolves

---

## 📊 QUICK REFERENCE: COMMON METRICS

| Metric Type | Examples | Use Case |
|-------------|----------|----------|
| **Accuracy** | Exact match, BLEU, ROUGE | Text similarity |
| **Retrieval** | Hit rate, MRR, NDCG | RAG systems |
| **Quality** | Factuality, Groundedness | Hallucination detection |
| **Safety** | Toxicity, Bias | Content safety |
| **Operational** | Latency, Throughput, Cost | Production monitoring |

```python
# Quick Example: Comprehensive Evaluation
def comprehensive_evaluate(app, test_data):
    return {
        "semantic_similarity": calculate_similarity(app, test_data),
        "factuality_score": check_grounding(app, test_data),
        "safety_score": run_safety_checks(app, test_data),
        "avg_latency": measure_latency(app, test_data),
        "cost_per_query": calculate_cost(app, test_data)
    }
```

---

## 02. Introduction to LLM Evaluations – Model Evals vs Application Evals (24:30)

## 🎯 1. Core Definition of LLM Evaluations

**LLM Evaluations are:** *Systematic, repeatable tests used to judge an LLM or an LLM-powered system against clear criteria.*

The creator breaks this definition into **3 key characteristics**:

| Characteristic | Meaning | Vibe Testing (Bad) vs Proper Evals (Good) |
| :--- | :--- | :--- |
| **Systematic** | You don't ask random questions from your head. You create proper **datasets** covering all edge cases (e.g., collecting 100 real user chats). | ❌ Asking 5 random questions → ✅ Testing on a structured dataset. |
| **Repeatable** | You can run the *exact same tests* even after changing the model, prompt, or retriever. This lets you compare Version 1 vs Version 2 objectively. | ❌ Testing once and forgetting it → ✅ Running the same test suite on every new version. |
| **Clear Criteria** | You define *specific rubrics* upfront (e.g., "Answer must be factual, simple, safe, and grounded in course material"). Without criteria, you are just vibe-testing. | ❌ "This answer feels right." → ✅ "Does this answer meet our 5 rubrics?" |

---

### 💻 Code Example: Systematic & Repeatable Testing

```python
# BAD: Vibe Testing (Not Systematic, Not Repeatable)
def vibe_test(chatbot):
    # Random questions from my head
    questions = ["What is AI?", "Tell me a joke"]
    for q in questions:
        print(chatbot.ask(q)) # Just look at it and say "seems fine"

# GOOD: Systematic & Repeatable (Uses a Dataset)
class LLMEvaluator:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset  # Contains 1000 curated questions

    def run_evaluation(self, chatbot_version):
        results = {}
        for test_case in self.test_dataset:
            response = chatbot_version.ask(test_case["question"])
            # Score against specific criteria
            results[test_case["id"]] = self.score_response(response, test_case)
        return results

    def score_response(self, response, test_case):
        score = 0
        if test_case["expected_keyword"] in response:
            score += 1
        # ... add more criteria checks
        return score / total_criteria

# Now I can run this same evaluator on Version 1, Version 2, and Version 3!
```

---

## ⚠️ 2. Big Clarification: Evals ≠ Metrics

**Common Misconception**: People think LLM Evals are just *metrics* like Accuracy, Precision, or Recall (similar to traditional Machine Learning).

**The Truth**: **LLM Evals are the ENTIRE TESTING SETUP**, not just the numbers.

- What are you testing? (Retriever? Final output?)
- What criteria are you using? (Factuality? Tone?)
- When are you testing? (Offline vs Production?)
- Which tools are you using? (RAGAS? Custom scripts?)

### 💻 Code Example: Evals as a "Testing Setup" (Not just a Metric)

```python
# This whole class IS the "Evaluation" (Setup), not just a single number.
class LLMApplicationEvalSetup:

    def __init__(self):
        # 1. WHAT: The component to test
        self.target_component = "RAG_Retriever" 
        
        # 2. DATASET: The data we test on
        self.test_queries = ["What is ML?", "How to code Python?"]
        
        # 3. CRITERIA: The rubrics
        self.criteria = ["Relevance", "Latency"]
        
        # 4. TOOLS: The library we use
        self.eval_tool = "RAGAS" 

    def run(self):
        # 5. EXECUTION: Offline run
        results = []
        for query in self.test_queries:
            retrieved_docs = retrieve(query)
            # Check if the docs are relevant
            score = check_relevance(retrieved_docs, query)
            results.append(score)
        # 6. METRICS: Finally, we output the average score
        average_score = sum(results) / len(results)
        return average_score
```

---

## ❓ 3. What Questions do Evals Answer?

The goal of an eval is **not just to give a score**. It answers **practical development questions**:

- Can this model be used for my specific application?
- Is this system good enough to ship to production?
- Did my new Prompt V2 actually improve over Prompt V1?
- Is the RAG answer *grounded* in the retrieved context?
- Is the agent completing the task correctly?
- Is the chatbot *safe* for real users?
- Is the system latency under control?

---

## 📂 4. The Two Types of LLM Evaluations

The creator splits LLM Evals into **two distinct categories** (Note: These are informal terms he coined for simplicity).

### A. Model Evals (Evaluating the Base LLM)
- **Goal**: To test the *capabilities* of the raw foundational model itself (e.g., GPT-4, Llama 3).
- **Who does this?**: Frontier labs (OpenAI, Google, Anthropic) when they release a new model.
- **Your Role (AI Engineer)**: You usually DON'T run these. You just need to **read and understand** them to decide which model to pick for your project.

**The 8 Core Capabilities tested in Model Evals**:
1. Reasoning (Step-by-step logic)
2. Knowledge (World knowledge up to cutoff date)
3. Basic Math
4. Coding
5. Instruction Following
6. Long Context Handling
7. Multimodal Understanding (Images, audio)
8. Tool Use

**Famous Benchmarks for each**:
- Knowledge/Reasoning → **MMLU**
- Math → **GSM8K**
- Coding → **SWE-bench**
- Instruction Following → **IF Eval**
- Long Context → **Needle in a Haystack**
- Multimodal → **MMMU**

---

### 💻 Code Example: Simulating a "Model Eval" (Reading a Benchmark)

```python
# As an AI Engineer, you don't train these. You READ these to pick a model.

# Simulated benchmark results from a leaderboard
model_benchmarks = {
    "GPT-4": {"MMLU": 86.4, "GSM8K": 92.0, "HumanEval": 85.0},
    "Claude-3": {"MMLU": 85.2, "GSM8K": 91.0, "HumanEval": 84.5},
    "Llama-3": {"MMLU": 81.3, "GSM8K": 79.0, "HumanEval": 82.0}
}

def choose_model_for_coding_task():
    # Since my app requires heavy coding, I pick the model with the best HumanEval score
    best_coding_model = max(model_benchmarks, key=lambda m: model_benchmarks[m]["HumanEval"])
    print(f"Selected {best_coding_model} for my project because it scores highest on coding.")
    # Output: Selected GPT-4
```

---

### B. Application Evals (Evaluating YOUR Full System) ⭐ **MAIN FOCUS**
- **Goal**: Test the entire system you built *around* the LLM.
- **Who does this?**: **You, the AI Engineer.** This is the most important topic in this playlist.
- **Why?**: The LLM is just **one component** (the "brain"). You also have Prompt design, Tools/APIs, Orchestration code, Guardrails, Output Parsers, Memory, Retrieval (in RAG), Vector DBs, and Monitoring.

**Smartphone Analogy**: 
- *Model Eval* = Checking the Snapdragon processor's benchmark score (Fast).
- *Application Eval* = Checking the entire phone: Camera quality, Battery life, Screen brightness, and OS smoothness. A fast processor doesn't guarantee a great phone!
- **Application Evals test the BEHAVIOR** of the whole product.

### 💻 Code Example: Application Eval (Testing a RAG Bot's Components)

```python
# You are testing the ENTIRE system (RAG), not just the base LLM.

class RAGApplicationEvaluator:
    
    def evaluate_full_system(self, rag_system, test_questions):
        # 1. SYSTEM LEVEL: Final answer quality
        for q in test_questions:
            response = rag_system.ask(q)
            assert_response_is_factual(response)  # Check hallucination
            assert_response_is_safe(response)     # Check safety
            
    def evaluate_individual_components(self, rag_system):
        # 2. COMPONENT LEVEL: Check the Retriever separately
        query = "What is the capital of France?"
        retrieved_docs = rag_system.retrieve(query)
        
        # Check if the Retriever actually pulled the correct document
        expected_doc = "Paris is the capital of France."
        if expected_doc not in retrieved_docs:
            print("WARNING: Retriever failed! Even if LLM answers well, the retriever is broken.")

        # 3. Check Latency (Operational metric)
        import time
        start = time.time()
        rag_system.ask(query)
        latency = time.time() - start
        assert latency < 2.0, f"Too slow! Latency is {latency}s" # Operational Eval
        
        print("All system and component checks passed!")
```

---

## 📝 5. Summary of the "What" & "Why"

- **Model Evals** = "Can the brain solve complex problems?" (You read these, rarely run them).
- **Application Evals** = "Does the entire phone product work smoothly, safely, and fast for my specific users?" (You run these ALL the time).

---

## 03. 


summaries this LLM Evaluation tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples