
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

## 03. How to Evaluate LLM Applications: The Complete Workflow (16:59)

## 🔄 Part 1: Quick Recap (From Videos 1 & 2)

Before starting the workflow, here is the recap:

1. **Why?** To avoid legal trouble, reputation damage, and expensive fines (Air Canada, Chevrolet, Lawyer cases).
2. **What?** Systematic, repeatable tests against clear criteria (not just random "vibe" testing).
3. **Types:**
   - **Model Evals**: Testing the raw base LLM (Done by OpenAI/Google). You just read these to pick a model.
   - **Application Evals**: Testing YOUR entire system (The AI Engineer's main job). **This is our focus.**

---

## 📧 Part 2: The Scenario (The Setup)

**You are an AI Engineer at Zomato.**

- **Problem**: Zomato gets thousands of customer emails daily. Manual replies are hard.
- **Your Goal**: Build a system that automatically reads an email, **classifies** it, and routes it to the correct internal team.
- **Categories (Labels)**:
  - `Billing` → Route to the Billing Team.
  - `Technical` → Route to the Tech Team.
  - `General` → Route to General Support.
- **The System**: It's just an LLM + a System Prompt telling the LLM to classify.

---

## 🛠️ Part 3: The 9-Step Application Evaluation Workflow

Here is the exact step-by-step process you will follow for *every* LLM app you build.

### Step 1: Define the Task and Target
Clearly state *what* you are testing.
- **Target**: The entire email routing system.
- **Task**: We need to evaluate if this system correctly classifies emails.

### Step 2: Define Success Criteria and Metrics
How do we know if the system is "good"?
- **Success Criteria**: Correct classification.
- **Metric**: **Accuracy** (e.g., out of 100 emails, how many did it route correctly?).

### Step 3: Build a "Golden" Evaluation Dataset
Create a test dataset with expected results (manually labeled by humans).
- You take past real emails from Zomato.
- You manually label them as `Billing`, `Technical`, or `General`.
- This is called a **Golden Dataset** (The "Answer Key" for your test).

### Step 4: Choose an Evaluation Method
Who will compare the LLM's answers to the Golden Dataset?
- **Option A: Automated (Code)** – Best for exact classification (like this case). Just write a Python script to compare labels.
- **Option B: Human** – Expensive. Use for subjective tasks (e.g., "Does this response sound empathetic?").
- **Option C: LLM-as-a-Judge** – Use a stronger LLM (e.g., GPT-4) to judge weaker models. Good for comparing long paragraphs where meaning is subjective.

### Step 5: Run the Model on the Dataset
Feed your Golden Dataset into your LLM system and let it generate predictions.

### Step 6: Evaluate & Analyze Results
Calculate the accuracy. Let’s say the result is **80%** (It misclassified 20 out of 100 emails). 
**Critical Step**: You don't just look at the number. You analyze *why* it failed. Is it confusing "Billing" with "Technical" often?

### Step 7: Improve the System (Optimization)
Based on the analysis, you improve the system:
- **Fix 1**: Tweak the System Prompt (add clearer definitions or examples).
- **Fix 2**: Upgrade to a better/smarter base LLM (if the current one is too dumb).

### Step 8: Iterative Evaluation Loop
You re-run the *exact same Golden Dataset* on the improved system.
- Version 1 scored 80%.
- You tweak the prompt → Re-run → Now it scores **90%**.
- You upgrade the model → Re-run → Now it scores **95%**.
- Loop continues until you are happy with the score.

### Step 9: Deploy & Production Monitoring
You deploy the system online. 
- **Monitoring**: Watch what happens in production.
- **Catching Failures**: A user complained to the Tech team, but it should have gone to Billing. The Tech team flags this mistake.
- **Feedback Loop**: You take that *specific failed email*, add it to your Golden Dataset, and run the evaluation loop again. 
- **Result**: Your dataset gets richer over time, and the system keeps improving forever.

---

## 💻 Code Examples for the Workflow

### 1. Building a Golden Dataset (Step 3)
```python
# This is your "Answer Key" (Golden Dataset)
golden_dataset = [
    {
        "email_text": "My credit card was charged twice this month.",
        "expected_label": "Billing"
    },
    {
        "email_text": "The app crashes every time I try to log in.",
        "expected_label": "Technical"
    },
    {
        "email_text": "What are your restaurant operating hours?",
        "expected_label": "General"
    }
]
# In reality, you will have 100 to 500 of these examples.
```

### 2. Automated Evaluation Method (Step 4 & 6)
```python
# This function acts as our "Automated Evaluator"
def calculate_accuracy(system_predictions, expected_labels):
    correct = 0
    total = len(expected_labels)
    
    for i in range(total):
        if system_predictions[i] == expected_labels[i]:
            correct += 1
            
    accuracy = (correct / total) * 100
    return accuracy

# Simulating Version 1 (80% accurate - 2 out of 10 wrong)
predictions_v1 = ["Billing", "General", "Billing"] # Let's say 3 examples, 2 correct
expected = ["Billing", "Technical", "Billing"] 
score = calculate_accuracy(predictions_v1, expected)
print(f"Version 1 Accuracy: {score}%") # Output: Version 1 Accuracy: 66.6% (if 2/3)
```

### 3. Iterative Improvement Loop (Step 7 & 8)
```python
# Simulating the iterative loop

def run_evaluation(dataset, llm_version):
    predictions = []
    for item in dataset:
        # In reality, you call your LLM API here with the prompt
        prediction = llm_version.classify(item["email_text"])
        predictions.append(prediction)
    
    # Compare predictions to dataset's "expected_label"
    return calculate_accuracy(predictions, [d["expected_label"] for d in dataset])

# Simulate Version 1 (Small, cheap model)
v1_accuracy = 80.0 

# You analyze failures, tweak the prompt (Version 1.1)
v1_1_accuracy = 88.0 

# You upgrade to GPT-4 (Version 2)
v2_accuracy = 95.0 

print(f"Prompt Tweaking: {v1_1_accuracy}%")
print(f"Model Upgrade: {v2_accuracy}%")
```

### 4. Production Feedback Loop (Step 9)
```python
# A failure happens in production
production_failure = {
    "email_text": "I didn't get my refund yet.",
    "actual_label": "Billing", # What it should have been
    "system_predicted": "General" # What the system wrongly gave
}

# You ADD this failure to your Golden Dataset for the next evaluation cycle
golden_dataset.append({
    "email_text": production_failure["email_text"],
    "expected_label": production_failure["actual_label"]
})

# Now next time you run the eval, the dataset is larger and catches more edge cases.
```

---

## ⚠️ Crucial Point: Multiple Evaluations per Application

The instructor emphasizes that **one application has MULTIPLE evaluations**, not just one.

For example, if you built a **RAG (Retrieval-Augmented Generation)** Chatbot, you would run:

| Evaluation Type | What it tests | Purpose |
| :--- | :--- | :--- |
| **Retriever Eval** | Does it fetch the correct documents? | To ensure the search engine works. |
| **Embedding Eval** | Is the embedding model capturing meaning? | To ensure vector search is accurate. |
| **Full Flow Eval** | Is the final LLM answer good? | To check hallucinations and coherence. |
| **Latency Eval** | How fast is the entire system? | To track operational performance and cost. |

---

## 📝 Final Summary Table (The Workflow)

| Step | Action | Example for this Email Classifier |
| :--- | :--- | :--- |
| **1** | Define Task | Evaluate the email routing system. |
| **2** | Define Metric | Accuracy score (%). |
| **3** | Golden Dataset | 100 real emails manually labeled as Billing/Tech/General. |
| **4** | Eval Method | Automated (Python code comparing strings). |
| **5** | Run System | Feed the 100 emails to the LLM. |
| **6** | Evaluate/Analyze | Accuracy is 80%. Figure out it's confusing similar texts. |
| **7** | Improve System | Edit the System Prompt to explain differences better. |
| **8** | Iterate | Re-run the same dataset → Accuracy jumps to 90%. |
| **9** | Deploy/Monitor | Put live. If users complain about wrong routing, add those emails back to the dataset and restart from Step 5. |

**Bottom Line**: Building an LLM app is easy (takes 5-10 minutes). Building a **production-ready, reliable LLM app** requires this strict, never-ending evaluation loop. 🚀

---

## 04. Why Your AI Application Needs Multiple Eval Pipelines? (28:05)

This lecture focus on a crucial practical question: *"If I build one LLM application, why do I need to write multiple different evaluation pipelines instead of just one?"*

There are **two major reasons**: **Multiple Failure Points** (components break individually or together) and **Multiple Risk Categories** (quality, safety, and operations). 

---

## 🧩 Reason 1: Multiple Failure Points (Component vs. Workflow)

An LLM application is not a single black box. It has multiple components. If Component A works and Component B works, **the whole system can still fail** because of how they interact.

### The RAG (Retrieval-Augmented Generation) Example

- **Components**:
  1. **Retriever**: Searches a Vector DB for relevant documents.
  2. **Generator (LLM)**: Reads the retrieved documents and answers the user.

#### The "K=5" Trap Scenario
- **User Question**: *"What is the duration of the ML course?"*
- **Retriever Setting**: `K=5` (fetches the top 5 documents).
- **Documents Fetched**:
  - Doc 1: Random info.
  - Doc 2: Random info.
  - Doc 3: Random info.
  - Doc 4: "The Python course duration is 6 weeks." (Irrelevant to ML).
  - Doc 5: **"The ML course duration is 8 weeks."** (Correct answer).

**Evaluation Check 1 (Retriever Eval)**: ✅ **PASS**. Why? Because the Retriever's job was to fetch 5 documents where *at least 1* is relevant. It found Doc 5. Good retriever.

**Evaluation Check 2 (Generator Eval)**: ✅ **PASS**. Why? The Generator is programmed to prioritize the *top-ranked* documents (Doc 1, 2, 3, 4) over lower ones. It sees "6 weeks" in Doc 4 and ignores Doc 5. It follows instructions perfectly.

**Actual System Output**: ❌ **"The ML course duration is 6 weeks."** (Wrong answer!).

**Conclusion**: Both components worked perfectly individually, but the **Workflow/Integration** failed because the correct document was buried at the bottom. 

**Solution**: You need a **Workflow-level Eval** that tests the Retriever + Generator *together*. It would catch this error and tell you to add a **Reranker** (which moves Doc 5 to the top).

---

### 💻 Code Example: Simulating the Component vs. Workflow Failure

```python
# Simulating the RAG components

class Retriever:
    def fetch(self, query, k=5):
        # Simulates fetching 5 docs. Correct answer is at index 4 (last).
        return ["Doc1_Random", "Doc2_Random", "Doc3_Random", "Doc4_Python_6weeks", "Doc5_ML_8weeks"]

class Generator:
    def generate(self, query, documents):
        # BAD LOGIC: Always picks the FIRST document (index 0) as the source of truth.
        # This is a "workflow" issue, not a component issue.
        first_doc = documents[0] 
        if "6weeks" in first_doc:
            return "6 weeks"
        elif "8weeks" in first_doc:
            return "8 weeks"
        return "Unknown"

# --- Running Individual Component Evals ---
retriever = Retriever()
docs = retriever.fetch("ML course duration")

# Component Eval 1: Retriever Test (Does it contain the right doc?)
assert "Doc5_ML_8weeks" in docs, "Retriever Failed!"  # ✅ PASS (it is in there)

# Component Eval 2: Generator Test (Does it hallucinate?)
generator = Generator()
response = generator.generate("ML course duration", docs)
# The generator DID NOT hallucinate. It used what it saw in the first doc.
print(f"Generator Output: {response}")  # Output: 6 weeks

# --- WORKFLOW-LEVEL EVAL (Testing them together) ---
# This eval runs the FULL pipeline and checks the FINAL answer.
expected_answer = "8 weeks"
if response != expected_answer:
    print(f"❌ WORKFLOW FAILED! Expected '{expected_answer}', got '{response}'.")
    print("💡 Fix: Add a Reranker to prioritize Doc5 before sending to Generator.")
# Output: ❌ WORKFLOW FAILED! Expected '8 weeks', got '6 weeks'.
```

---

### The 3 Levels Where Failures Happen
Even if you fix the Workflow, you still need a third level.

| Level | What Breaks | Example | Eval Needed |
| :--- | :--- | :--- | :--- |
| **1. Component** | Individual pieces | Retriever fetches garbage / Generator hallucinates. | Component-specific Eval. |
| **2. Workflow** | Interaction between pieces | Correct doc is buried, logic prioritizes wrong docs. | Integration/Flow Eval. |
| **3. Application** | Overall system performance | The pipeline returns the *correct* answer, but takes **10 seconds** to reply. | Operational/Latency Eval. |

---

## ⚠️ Reason 2: Multiple Risk Categories (Quality, Safety, Ops)

Even if your system is technically "correct", it can still fail in different **dimensions**. 

The instructor splits all risks into **3 Main Pillars**. You need a separate eval pipeline for each pillar.

1. **Application Quality**: "Does it do the actual job well?"
   - Correctness, Relevance, Completeness, Groundedness/Faithfulness, Instruction Following.
2. **Safety**: "Is it harmful or leaking data?"
   - Toxicity, Bias, PII (Private Data) Leaks, Jailbreak Resistance.
3. **Operations**: "Is it fast, cheap, and reliable?"
   - Latency, Cost per request, Token efficiency, Error rates under load.

### ⚠️ Important Insight:
- The **SAME component** needs to be checked across different risk categories.
- *Example*: Your Retriever fetches relevant documents (Quality check = ✅). But it takes 5 seconds to do it (Operations check = ❌). You need a Latency Eval for the Retriever *and* a Quality Eval for the Retriever. They are two different pipelines.

---

### 💻 Code Example: Evaluating the Same Component for Different Risks

```python
# Imagine we have a RAG system's Retriever component.

class Retriever:
    def fetch(self, query):
        # Simulating a slow but accurate retriever
        import time
        time.sleep(5)  # Takes 5 seconds!
        return ["Relevant Document 1", "Relevant Document 2"]

# --- Eval 1: QUALITY RISK (Does it fetch relevant docs?) ---
def evaluate_retriever_quality(retriever, query, expected_docs):
    fetched = retriever.fetch(query)
    # Check if the expected docs are in the fetched list
    score = len(set(fetched) & set(expected_docs)) / len(expected_docs)
    return score  # If score is 1.0, Quality is good.

# --- Eval 2: OPERATIONAL RISK (Does it fetch fast enough?) ---
def evaluate_retriever_latency(retriever, query, max_allowed_seconds=2.0):
    import time
    start = time.time()
    retriever.fetch(query)
    latency = time.time() - start
    
    if latency > max_allowed_seconds:
        return f"❌ Ops Failed! Latency is {latency}s, exceeding {max_allowed_seconds}s."
    else:
        return f"✅ Ops Passed! Latency is {latency}s."

# --- Running the evaluations ---
my_retriever = Retriever()
query = "ML course duration"
expected = ["Relevant Document 1"]

quality_score = evaluate_retriever_quality(my_retriever, query, expected)
latency_result = evaluate_retriever_latency(my_retriever, query, 2.0)

print(f"Quality Score: {quality_score * 100}%")  # Output: 100% (Good!)
print(latency_result)  # Output: ❌ Ops Failed! Latency is 5.0s, exceeding 2.0s.
# Conclusion: We need to run a different Eval (Latency) to catch this specific risk!
```

---

### 📊 Specific Metrics by Application Type (The Instructor's Table)

Here are the risks you will evaluate based on what you are building:

| Application Type | Risks to Evaluate (Quality Pillar) |
| :--- | :--- |
| **General LLM App** (e.g., Summarizer) | Correctness, Relevance, Completeness, Instruction Following. |
| **RAG App** (Retrieval + Gen) | Context Relevance (Retriever), Groundedness/Faithfulness, Citation Accuracy. |
| **Agent App** (Uses Tools) | Tool Selection, Parameter Correctness, Task Completion, Error Recovery. |
| **Multi-Turn Chatbot** | Context Retention (long-term memory), Clarification Behavior. |

| Safety Pillar | Operational Pillar |
| :--- | :--- |
| Toxicity (Hate speech) | Latency (Time to first token) |
| Harmful Content (Weapons, Self-harm) | Cost per request (Token usage) |
| Bias (Gender/Race stereotyping) | Token Efficiency |
| PII Leaks (Phone numbers, emails) | Error Rate & Performance under Load |
| Prompt Injection / Jailbreak Resistance | Throughput |

---

## 📝 Final Summary of the Session

| Key Point | Explanation |
| :--- | :--- |
| **One App ≠ One Eval** | A single application needs multiple evaluation pipelines running simultaneously. |
| **Reason 1: Failures** | Components can work individually, but fail when combined (e.g., RAG's K=5 trap). You need Component, Workflow, AND Application-level evals. |
| **Reason 2: Risks** | A system can be factually correct but unsafe, slow, or expensive. You need Quality, Safety, AND Operational evals. |
| **Actionable Takeaway** | When you build your next LLM app, don't just write one test script. Write a script for **Retriever Quality**, one for **Answer Faithfulness**, one for **Latency**, and one for **PII Leakage**. |

**Bottom Line**: Production-grade AI engineering is not about building a cool demo. It is about building multiple guardrails (evals) to catch every possible way your system can fail, whether it's a logical pipeline flaw or a security vulnerability. 🚀

---

## 05. LLM Eval Methods | LLM-as-a-Judge | Reference Based Evals Vs Reference Free Evals (53:09)

This lecture focusing on the core question: *"Who or what actually performs the comparison in an LLM evaluation pipeline?"* 

The instructor defines the **3 core Evaluation Methods** (Programmatic, Human, and LLM-as-a-Judge), walks through detailed real-world examples for each, and ends with a key conceptual distinction (Reference-Based vs. Reference-Free).

---

## 🎯 Part 1: What is an "LLM Evaluation Method"?

**Definition**: It is the **mechanism/executor** that takes an LLM output and produces a judgment about it. 

**The 3 Core Methods**:
1. **Programmatic (Deterministic)**: A Python script/code runs the comparison.
2. **Human**: A human expert manually reviews and scores the output.
3. **Model-Graded (LLM as a Judge)**: Another (usually stronger) LLM runs the comparison.

---

## 💻 Method 1: Programmatic / Deterministic Evaluation

**When to use**: When the task has a **clearly defined correct answer** that code can easily compare (e.g., classification, retrieval accuracy).

**Example Scenario**: Evaluating a **Retriever** in a RAG system for a CampusX chatbot. 
The Retriever's job is to fetch relevant documents from a vector database for a given query.

### Step-by-Step Workflow (Programmatic)

1. **Task & Target**: Evaluate the Retriever component.
2. **Success Criteria & Metric**: **Recall@K**.
   - *Formula*: (Number of relevant documents retrieved in the top K) / (Total number of relevant documents that exist).
3. **Golden Dataset**: A set of 50-100 test queries. For each query, a human expert manually tagged which document IDs contain the correct answer (e.g., Query 1 → Doc 1001; Query 2 → Docs 1001 & 1003).
4. **Execution (Programmatic)**:
   - Set `K=5`.
   - Feed each query to the Retriever → It returns 5 document IDs.
   - A Python script compares the Retrieved IDs vs. the Golden IDs to calculate Recall.
   - Average the Recall across all queries.
5. **Analysis & Improvement**: If Recall is low (e.g., 67%), you can improve by:
   - Changing the Embedding Model.
   - Increasing `K` (e.g., from 5 to 10).
   - Adding a Reranker to push relevant docs higher.

### 💻 Code Example: Programmatic Recall@K

```python
# Golden Dataset (Pre-tagged by a Human Expert)
golden_data = {
    "Q1: ML course prerequisites": {"relevant_docs": ["Doc_1001"]},
    "Q2: ML course duration": {"relevant_docs": ["Doc_1001", "Doc_1003"]},
}

# Simulated Retriever output for Q2 with K=5
retrieved_docs_for_Q2 = ["Doc_1001", "Doc_1002", "Doc_1004", "Doc_1005", "Doc_1006"]

# --- Programmatic Evaluation (The Python Script) ---
def calculate_recall_at_k(retrieved, relevant, k=5):
    # Out of all relevant docs, how many did we fetch in our top-k?
    if not relevant:
        return 1.0  # Edge case
    fetched_relevant = set(retrieved) & set(relevant)
    return len(fetched_relevant) / len(relevant)

# Run the script
recall = calculate_recall_at_k(retrieved_docs_for_Q2, golden_data["Q2: ML course duration"]["relevant_docs"])
print(f"Recall@5 for Q2: {recall * 100}%") 
# Output: 50% (Fetched Doc_1001, missed Doc_1003)

# Average over the entire dataset programmatically
# This is cheap, fast, and repeatable! ✅
```

---

## 🧑‍🏫 Method 2: Human-Based Evaluation

**When to use**: When the output is **subjective** (e.g., "Is this response helpful?", "Does this have a good tone?"). Code cannot easily measure these.

**Example Scenario**: Evaluating a general CampusX chatbot's **"Helpfulness"** on a scale of 1 to 5.

### Step-by-Step Workflow (Human)

1. **Task & Target**: Evaluate the entire chatbot's helpfulness.
2. **Success Criteria**: A **Rubric** (1 = Not helpful, 3 = Partially helpful, 5 = Perfectly accurate, complete, and has the right tone).
3. **Golden Dataset**: A list of 50-100 diverse questions (no correct answers provided here).
4. **Execution (Human)**:
   - Feed each question to the chatbot → Get an answer.
   - Give the question and answer to a Human Grader.
   - The Human reads the rubric and assigns a score (1-5).
   - (Optional but recommended) Use **multiple graders**. If they disagree a lot, your rubric is ambiguous. If they agree, the rubric is clear.
5. **Cost vs. Reliability Trade-off**: 
   - **Pro**: Highly reliable judgment.
   - **Con**: Very expensive and slow to scale.

### Other Types of Human Evals Mentioned:
- **Red Teaming**: Humans intentionally try to break the system (jailbreak, prompt injection).
- **A/B Testing**: Users rate two versions in production to choose the better one.
- **Human-in-the-Loop**: A human reviews and corrects outputs when the system is uncertain.

### 💻 Code Example: Simulating Human Grading & Agreement

```python
# Simulating 2 Human graders scoring a chatbot's answer (1-5 scale)
grader_A_scores = [5, 4, 3, 2, 5]  # Scores for 5 questions
grader_B_scores = [4, 4, 2, 1, 5]  # Scores for the same 5 questions

# Calculate average helpfulness score for the system (Human evals)
average_score = sum(grader_A_scores) / len(grader_A_scores)
print(f"System Helpfulness Score (Grader A): {average_score}/5")

# Calculate agreement (disagreement metric)
def calculate_disagreement(list_a, list_b):
    diff_sum = sum(abs(a - b) for a, b in zip(list_a, list_b))
    return diff_sum / len(list_a)

disagreement = calculate_disagreement(grader_A_scores, grader_B_scores)
print(f"Average Disagreement between graders: {disagreement:.2f} points")
# If disagreement is high (e.g., 1.5), the rubric needs to be refined! ✅
```

---

## 🤖 Method 3: Model-Graded Evaluation (LLM as a Judge)

**When to use**: When the task is subjective (like human eval) but you need to scale it cheaply. This is the **most popular method** in production today.

**Example Scenario**: Building an automated platform to evaluate **UPSC Mains (subjective) exam papers** for thousands of students, where hiring human experts would be too expensive.

### Step-by-Step Workflow (LLM as a Judge)

1. **Task & Target**: Evaluate the entire system that grades UPSC Mains answers.
2. **Success Criteria**: The LLM grader must match the human expert's grading closely.
3. **Golden Dataset**:
   - Create a **Rubric** for each question (e.g., For a 15-mark question: check for definition, mechanisms, examples, balanced conclusion).
   - A Human Expert grades **only 50 sample answers** using this rubric (these become the "Ground Truth" scores).
4. **Execution (LLM)**:
   - Write a detailed **System Prompt** telling the LLM: "You are a UPSC examiner. You have this rubric. Grade the following student answer. Give marks and a reasoning justification."
   - Run the LLM on the same 50 answers.
5. **Evaluate the Judge**: Compare Human Scores vs. LLM Scores using **Mean Absolute Error (MAE)**.
   - *Formula*: `MAE = (Sum of |Human Score - LLM Score|) / (Number of Questions)`.
   - If MAE is 2.3, it means the LLM is off by an average of 2.3 marks. The goal is to bring this number down to 0 (perfect match).

### 💻 Code Example: LLM as a Judge & MAE Calculation

```python
# Golden Dataset: Human scores for 50 student answers (Ground Truth)
human_scores = [13, 4, 8, 10]  # 4 sample answers (Max marks: 15)

# LLM Judge outputs (after we prompted it with the rubric)
llm_scores = [12, 8, 8, 9]     # Simulated LLM predictions

# --- Calculating Performance (Metric for the Judge) ---
def calculate_mae(human, llm):
    errors = [abs(h - l) for h, l in zip(human, llm)]
    return sum(errors) / len(errors)

mae = calculate_mae(human_scores, llm_scores)
print(f"Mean Absolute Error (MAE): {mae} marks")
# Output: 2.0 marks
# Interpretation: On average, the LLM is 2 marks off from the human expert.

# We can now iterate:
# 1. Improve the system prompt.
# 2. Upgrade to a stronger LLM (GPT-4 instead of GPT-3.5).
# 3. Rerun the MAE to see if it drops closer to 0.
```

---

## 📑 Part 2: Reference-Based vs. Reference-Free Evaluations

This is a conceptual distinction independent of the 3 methods above. It simply asks: **Does your Golden Dataset contain the "Correct Answer" or not?**

| Type | Definition | Does Dataset have the right answer? | Examples from the video |
| :--- | :--- | :--- | :--- |
| **Reference-Based** | You compare the LLM output against a **pre-defined correct answer**. | ✅ YES | **Retriever Eval** (We know Doc 1001 is correct). <br> **UPSC LLM Judge** (We have the Human Expert's scores as the reference to compare against). |
| **Reference-Free** | You judge the quality directly based on a **rubric/criteria**, without a single correct answer. | ❌ NO | **Human Helpfulness Eval** (No pre-defined correct answer; the human just rates it 1-5 based on feeling and rubric). |

> **Simple Check**: Look at your golden dataset. If it has an "Expected Answer" column → **Reference-Based**. If it just has a "Question" column and you rely on a judge (human or LLM) to interpret quality → **Reference-Free**.

---

## 📝 Summary Table of Methods

| Method | Executor | Best For | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Programmatic** | Code (Python) | Deterministic tasks (Classification, Retrieval). | Cheap, Fast, 100% Repeatable. | Cannot handle subjectivity. |
| **Human** | Human Experts | Subjective tasks (Helpfulness, Tone, Red Teaming). | Highly Reliable, Great for complex judgment. | Expensive, Slow, Hard to scale. |
| **LLM-as-a-Judge** | Another LLM | Subjective tasks that need scale (Grading essays, open-ended QA). | Cheap at scale, Fast, Mimics human logic. | Can be biased towards its own style, Costly for small tests. |

---

## 🚀 The Final Mindset Shift

The instructor emphasizes a crucial point at the end: 
> *"Building an LLM app is the easy part. The difficult part is making sure it works correctly **every time, everywhere**."*

Moving forward, you are not just a "builder". You are a **Production AI Engineer** who thinks about:
1. **Where** can this system fail? (Multiple Failure Points).
2. **Who** should judge if it's working? (Programmatic, Human, or LLM?).
3. **How** do we know it's good enough to deploy? (MAE < Threshold, Recall > 90%).

---

## 06. Offline Evals Vs Online Evals (01:21:08)

summaries this LLM Evaluation tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples

- [Notes](https://onedrive.live.com/personal/85452F67DAA1111C/_layouts/15/Doc.aspx?sourcedoc={90714588-0955-47bc-bf9e-176879959e0d}&action=view&redeem=aHR0cHM6Ly8xZHJ2Lm1zL28vYy84NTQ1MkY2N0RBQTExMTFDL0lnQ0lSWEdRVlFtOFI3LWVGMmg1bFo0TkFheWVYXzlSM1Y0WEhERG1zWFlNbnJr&wd=target%281.%20Introduction%20to%20LLM%20Evals.one%7Ca35dbc27-08ab-4743-b3b0-6b36c29acfd5%2FCourse%20Outline%7C165aacba-2851-e54b-bf9f-885a1a42b9ee%2F%29&wdorigin=NavigationUrl)