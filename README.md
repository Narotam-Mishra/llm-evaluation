
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

The lecture explains why you cannot rely solely on pre-launch testing, using a powerful UPSC exam grader example to illustrate the difference between **"Correctness"** and **"Normalcy"**.

---

## 🔄 Part 1: Quick Recap of Previous Sessions

Before starting today’s topic, here is a recap of what we have covered so far:

1. **Why Evals?** To avoid legal troubles, reputation damage, and hallucinations.
2. **What are Evals?** Systematic, repeatable tests against clear criteria.
3. **Types of Evals:** Model Evals (done by labs) vs. Application Evals (done by you).
4. **Eval Pipeline:** Task Definition → Dataset → Run → Analyze → Improve → Deploy.
5. **Multiple Pipelines:** One app needs multiple evals because of **multiple failure points** (Component, Workflow, Application) and **multiple risk categories** (Quality, Safety, Operations).
6. **Eval Methods:** Programmatic (code), Human, and LLM-as-a-Judge.

Now, we move to the next crucial topic: **Offline vs. Online Evaluations**.

---

## 🏠 Part 2: Offline Evaluations (Pre-Deployment Testing)

**Definition**: These are the evaluations you run on your application **BEFORE** you deploy it to production (staging/testing environment).

**Key Characteristic**: You have a **fixed "Golden Dataset"** with known correct answers. You run tests against this dataset to check **Correctness**.

### The 3 Massive Benefits of Offline Evals

1. **Pre-Release Testing / Release Gating**:
   - You can set a threshold (e.g., 95% accuracy).
   - If the score passes the threshold, the code is automatically deployed (via CI/CD).
   - If it fails, deployment is blocked, and the previous version stays live.

2. **Version Comparison**:
   - Unsure whether to use GPT-4 or Claude 3 for your app?
   - Run the *exact same* offline eval on both versions.
   - Compare the scores objectively and pick the winner.

3. **Regression Testing**:
   - *Definition*: Testing to ensure a new change (e.g., prompting the chatbot to be "kinder") doesn't break existing functionality (e.g., accidentally giving vague price estimates).
   - Your golden dataset has different types of questions (Refund, Pricing, Curriculum). Run the eval before and after the change. If Pricing accuracy drops, a regression has occurred—don't deploy!

### 💻 Code Example: Offline Eval Benefits

```python
# 1. RELEASE GATING (CI/CD Simulation)
offline_score = run_offline_eval(golden_dataset, model_v2)
if offline_score >= 95.0:
    print("✅ Passed! Deploying to production...")
    trigger_deployment()
else:
    print(f"❌ Failed! Score {offline_score}% < 95%. Rolling back.")
    rollback_to_previous_version()

# 2. VERSION COMPARISON (Model A vs Model B)
def compare_versions(dataset, model_a, model_b):
    score_a = run_eval(dataset, model_a)
    score_b = run_eval(dataset, model_b)
    if score_a > score_b:
        print(f"Model A wins! ({score_a}% vs {score_b}%)")
    else:
        print(f"Model B wins! ({score_b}% vs {score_a}%)")
    return max(score_a, score_b)

# 3. REGRESSION TESTING (Checking if fixing one thing breaks another)
# Dataset contains 30 Refund questions, 30 Pricing questions, 30 Curriculum questions.
def run_regression_test(old_version, new_version, dataset):
    old_scores = run_eval_by_category(old_version, dataset)
    new_scores = run_eval_by_category(new_version, dataset)
    
    for category in old_scores.keys():
        if new_scores[category] < old_scores[category] * 0.95: # 5% drop threshold
            print(f"⚠️ Regression detected in {category}! Score dropped from {old_scores[category]}% to {new_scores[category]}%.")
            return False # Do not deploy!
    print("✅ No regression. Safe to deploy.")
    return True
```

---

## 🚨 Part 3: The 3 Major Risks in Production (Why Offline Eval is NOT enough)

Even if your offline evals pass with 100% accuracy, once you deploy, you face entirely new risks:

1. **Unanticipated User Inputs**:
   - Users will ask questions you *never* added to your golden dataset (mixed Hindi/English, ambiguous half-questions, angry rants, adversarial prompt injections).
   - You cannot predict every possible user query.

2. **Emergent Systematic Failures**:
   - **Concurrency/Latency**: Your system works fine with 10 users, but crashes or slows down with 10,000 concurrent users.
   - **Bias**: A bias (e.g., against non-technical background users) only becomes statistically visible across thousands of conversations. You can't catch this with a few hundred test cases.

3. **Data/Concept Drift**:
   - Your business changes over time (prices change, course curriculums update, policies change).
   - Your Golden Dataset was created based on *today's* data. One year later, the data distribution is completely different.
   - Your offline eval will still show good scores, but in production, users will hate the chatbot because it gives outdated information.

---

## 🌐 Part 4: Online Evaluations (Post-Deployment Monitoring)

**Definition**: Evaluations run **AFTER** deployment, on **live production traffic**, as real users interact with the system.

**Key Characteristic**: **No Answer Key exists**. You do not know the "correct" answer for the live questions users are asking right now.

**Core Difference**:
- **Offline Eval** checks **Correctness** (Is the output right compared to the known correct answer?).
- **Online Eval** checks **Normalcy** (Is the system behaving *normally* compared to a historical baseline?).

### The UPSC Grader Example Explained

**The System**: An LLM that grades UPSC Mains (subjective) exam papers like a human expert.

- **Offline Phase (Checking Correctness)**:
  - You took 100 old student answers.
  - A Human Expert graded them (Ground Truth).
  - Your LLM graded the same 100 answers.
  - You calculated **Mean Absolute Error (MAE)** between Human and LLM scores.
  - If MAE is low (e.g., 1.2 marks), you assume it's "Correct" and deploy.

- **Online Phase (Checking Normalcy)**:
  - Now the system is live. 10,000 new students write exams today.
  - You **do NOT** have human scores for these 10,000 answers to compare against.
  - So, you cannot check *Correctness*.
  - Instead, you plot the **Distribution of Scores** (bell curve) for Week 1.
  - You record this as your **Baseline Normalcy** (e.g., Mean = 500/1000, Std Dev = 100).
  - In Week 2, you plot the distribution again. Suddenly, the mean jumps to 800/1000.
  - **Alert!** The system is not running *normally* anymore. Something changed (maybe a prompt update made it too lenient, or a bug broke the grading logic). You investigate immediately.

### 💻 Code Example: Online Monitoring of Score Distributions

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating weekly score distributions from the UPSC grader system

# Week 1 Baseline (Normal)
week1_scores = np.random.normal(loc=500, scale=100, size=1000)  # Mean 500
baseline_mean = np.mean(week1_scores)
baseline_std = np.std(week1_scores)
print(f"Week 1 Baseline: Mean = {baseline_mean:.1f}, Std = {baseline_std:.1f}")

# Week 2 Monitoring (Suddenly, the system starts giving higher marks)
week2_scores = np.random.normal(loc=800, scale=90, size=1000) # Mean jumped to 800

current_mean = np.mean(week2_scores)
current_std = np.std(week2_scores)

# Check for "Normalcy" using a threshold (e.g., 10% deviation from baseline)
if abs(current_mean - baseline_mean) > (0.1 * baseline_mean):
    print(f"🚨 ALERT: Mean shifted from {baseline_mean:.1f} to {current_mean:.1f}!")
    print("🔍 Investigate: System is behaving ABNORMALLY (Potential Drift/Bug).")
else:
    print("✅ System is running NORMALLY.")

# Plotting would show the bell curve shifting to the right.
```

---

## 📊 Part 5: Side-by-Side Comparison Matrix

| Feature | Offline Evaluation | Online Evaluation |
| :--- | :--- | :--- |
| **Timing** | **Before** Deployment. | **After** Deployment. |
| **Data** | **Fixed Golden Dataset** (pre-collected). | **Live Production Traffic** (real-time data). |
| **Answer Key** | ✅ **Exists** (You have the correct answers). | ❌ **Does not exist** (No correct answer available live). |
| **Inputs** | You only test **anticipated** edge cases. | Users can throw **any unanticipated** input. |
| **What it Checks** | **Correctness** (Is the answer right?). | **Normalcy** (Is the behavior normal compared to baseline?). |
| **Catch** | Catches **Regressions** (Pre-launch). | Catches **Drift, Emergent Bugs, and Surprises** (Post-launch). |
| **Cost/Speed** | **Cheap & Fast** (small dataset). | **Expensive** (requires sampling large data). Often uses sampling (e.g., monitor only 1,000 out of 50,000 daily conversations). |

---

## 🔧 Part 6: Alternative Online Signals (When you don't have a baseline)

If you cannot compare distributions, you can use **User Signals** as a proxy for correctness:

- **Thumbs Up / Thumbs Down**: If you see a sudden spike in "Thumbs Down" in the last hour, something is wrong with your chatbot's correctness.
- **Escalations**: If your customer support team suddenly gets a flood of complaints about wrong answers, your system has lost correctness, even if the score distribution looks normal.

### 💻 Code Example: Monitoring User Feedback Signals

```python
# Tracking user feedback over time
feedback_log = {
    "10:00 AM": {"thumbs_up": 50, "thumbs_down": 2},
    "11:00 AM": {"thumbs_up": 60, "thumbs_down": 3},
    "12:00 PM": {"thumbs_up": 45, "thumbs_down": 40}, # Spike in downvotes!
}

threshold = 20  # Acceptable downvotes per hour

def monitor_live_feedback(log):
    for hour, data in log.items():
        if data["thumbs_down"] > threshold:
            print(f"⚠️ ALERT: Downvote spike at {hour}! {data['thumbs_down']} downvotes.")
            print("💡 This indicates users are likely receiving incorrect answers.")
            trigger_investigation()

monitor_live_feedback(feedback_log)
```

---

## 📝 Final Summary

| Concept | Definition | Example |
| :--- | :--- | :--- |
| **Offline Eval** | Testing done **before launch** with a golden dataset to check **Correctness**. | Comparing your LLM's score to a Human Expert's score on 100 questions. |
| **Online Eval** | Monitoring done **after launch** on live traffic without an answer key to check **Normalcy**. | Plotting the weekly bell curve of scores to see if the distribution suddenly shifts. |
| **Complementary** | Offline and Online Evals are **not rivals**. You **need both**. Offline prevents bad launches; Online prevents silent death in production due to drift or unforeseen user behavior. |

**Bottom Line**: Offline evals tell you, *"Is this system technically correct to launch?"* Online evals tell you, *"Is this system still working correctly for real users right now?"* You cannot skip either if you want a production-grade AI application. 🚀

---

## Offline Vs Online EVals (Contd...)

The part of tutorial explains how to monitor a live LLM application, from logging conversations to setting up alerts, and emphasizes the continuous feedback loop between online failures and offline testing.

## 📌 Part 1: Step 1 – Non-Blocking Durable Logging

**Definition**: Before you can evaluate anything in production, you must **record (log) everything** that happens during every user conversation.

**What to Log (The Data Schema)**:
- `Conversation ID`, `Turn ID`, `User ID`, `Session ID`, `Timestamp`.
- User's question (Prompt).
- Retrieved Context (if using RAG).
- LLM's final output (Response).
- Operational metadata: Latency (ms), Prompt Tokens, Completion Tokens, Total Cost, Error Status Codes.
- User signals: Thumbs Up/Down, Escalations (user asking for a human), repeated rephrasing of the same question.

**4 Key Engineering Properties of Logging**:
1. **Non-Blocking**: Logging should NOT slow down the user's experience. Use async/background processes.
2. **Durable & Queryable**: Store logs in a proper data warehouse/observability tool (like **LangSmith**) so you can search/filter them later.
3. **Late Signal Attachments**: Sometimes user feedback (like an angry email) comes *hours later*. You must be able to attach that feedback to the correct `Conversation ID` later.
4. **PII Masking**: Before storing, automatically mask Personal Identifiable Information (phone numbers, credit cards, email IDs) to maintain privacy and security.

### 💻 Code Example: Non-Blocking Async Logging

```python
import asyncio
import uuid
from datetime import datetime

# Simulated logging function (writes to a database/observability tool)
async def log_conversation_async(user_query, llm_response, metadata):
    # Simulate a small network/database write delay (0.1 seconds)
    await asyncio.sleep(0.1) 
    log_entry = {
        "conversation_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "query": user_query,
        "response": llm_response,
        "latency_ms": metadata.get("latency", 0),
        "cost": metadata.get("cost", 0.0),
        "status": metadata.get("status", 200)
        # Note: PII masking would happen here before saving.
    }
    print(f"Logged: {log_entry['conversation_id']}")
    return log_entry

# Main chatbot handler (Non-blocking behavior)
async def handle_user_query(user_input):
    # 1. Start the LLM call (simulate)
    start_time = datetime.now()
    llm_output = f"Response to: {user_input}"  # Simulate LLM
    latency = (datetime.now() - start_time).total_seconds()
    
    # 2. Fire the logging in the background (does NOT await fully, or uses create_task)
    # The user gets the response IMMEDIATELY without waiting for the log to finish.
    asyncio.create_task(log_conversation_async(
        user_input, 
        llm_output, 
        {"latency": latency, "cost": 0.002}
    ))
    
    # 3. Return response to user instantly.
    return llm_output

# Simulating a user request
response = asyncio.run(handle_user_query("What is the price?"))
print(f"User sees: {response}") 
# Output: Logging happens in background, user doesn't wait for it.
```

---

## 📊 Part 2: Two Types of Signals (Captured vs. Computed)

Once you log the data, you need to extract meaningful metrics. They fall into two categories:

1. **Captured Signals (No Computation)**:
   - Directly recorded during the conversation. No extra processing needed.
   - Examples: Latency, Total Cost, Token usage, Thumbs Up/Down, Error Codes.
2. **Computed Signals (Needs Computation)**:
   - You must calculate these by running an evaluator (usually an **LLM-as-a-Judge**) on the logged data.
   - Examples: Faithfulness (groundedness), Answer Relevancy, Toxicity, Bias, Hallucination.

### 💻 Code Example: Differentiating Signals

```python
# Simulating a logged conversation trace from LangSmith
trace = {
    "question": "Tell me about Python.",
    "response": "Python is a high-level language.",
    "latency": 1.2,           # 🟢 CAPTURED (Just store it)
    "total_cost": 0.004,      # 🟢 CAPTURED (Just store it)
    "thumbs_up": None,        # 🟢 CAPTURED (Wait for user to click)
}

# 🟡 COMPUTED SIGNAL: Hallucination Score (Needs an LLM Judge)
def compute_hallucination_score(question, response, retrieved_context):
    # In reality, we call GPT-4/GPT-3.5 with a special rubric.
    # Simulate a score between 0 and 1.
    if "Python" in response and "programming" in retrieved_context:
        return 0.95  # Very faithful
    else:
        return 0.20  # Likely hallucinated

# We only compute this OFFLINE/ASYNC after the trace is logged.
hallucination_score = compute_hallucination_score(
    trace["question"], 
    trace["response"], 
    "Python is a programming language." 
)
print(f"Computed Hallucination Score: {hallucination_score}")
```

---

## 🔄 Part 3: Pipeline for Captured Signals (Simple Flow)

For **Captured Signals**, the flow is straightforward:

1. **Logging** → Store the raw metric (e.g., `latency = 1.2s`).
2. **Dashboarding** → Aggregate these metrics over time (e.g., Average latency for the last 1 hour) and visualize them on a graph.
3. **Alerting** → If the aggregated metric crosses a threshold (e.g., Average latency > 3 seconds for 5 minutes), trigger an alert (Slack, Email, PagerDuty).

---

## 🤖 Part 4: Pipeline for Computed Signals (The Core Challenge)

For **Computed Signals**, you cannot afford to run an LLM-as-a-Judge on every single conversation (too expensive). You must **sample**.

**The Full Flow**:
1. **Logging** (Store all 5,000 daily conversations).
2. **Sampling** (Select a subset, e.g., 1,000 out of 5,000).
3. **Evaluation** (Run your LLM-as-a-Judge evaluator *only* on the sampled conversations to compute metrics like Hallucination Rate).
4. **Dashboarding** (Visualize the computed metric over time).
5. **Alerting** (Trigger alerts if the metric deviates from the baseline).

### 🎯 Why Stratified Sampling > Random Sampling
Randomly picking 1,000 out of 5,000 conversations is risky. You might miss all the problematic ones! 
**Stratified Sampling**: You categorize conversations first and sample more heavily from "high-risk" categories:
- Conversations with **Thumbs Down**.
- Conversations that ended in **Escalation** (user asked for a human).
- Conversations involving **Money/Refunds** (high business impact).

### 💻 Code Example: Stratified Sampling Strategy

```python
import random

# Simulated daily conversations (logged traces)
daily_conversations = [
    {"id": 1, "category": "normal", "feedback": "thumbs_up"},
    {"id": 2, "category": "refund", "feedback": "thumbs_down"}, # High risk
    {"id": 3, "category": "normal", "feedback": None},
    # ... imagine 5000 of these
]

# Function to perform stratified sampling
def stratified_sample(conversations, total_sample_size=1000):
    # Separate into high-risk and low-risk
    high_risk = [c for c in conversations if c["category"] in ["refund", "payment"] or c["feedback"] == "thumbs_down"]
    low_risk = [c for c in conversations if c not in high_risk]
    
    # Sample 70% from high-risk and 30% from low-risk (overweight high-risk)
    sample_size_high = int(total_sample_size * 0.7)
    sample_size_low = total_sample_size - sample_size_high
    
    sampled_high = random.sample(high_risk, min(len(high_risk), sample_size_high))
    sampled_low = random.sample(low_risk, min(len(low_risk), sample_size_low))
    
    return sampled_high + sampled_low

# Run the evaluator ONLY on this biased sample to catch more bugs per dollar.
sampled_traces = stratified_sample(daily_conversations, 1000)
print(f"Evaluating {len(sampled_traces)} traces (focused on risky ones)")
```

---

## 🛠️ Part 5: Platform Demo (LangSmith Features)

The instructor shows LangSmith as a complete platform where:

- **Traces** = Logged conversations.
- **Evaluators** = Pre-built templates for hallucination, PII leakage, prompt injection, toxicity, etc., all using **LLM-as-a-Judge**.
- **Key Differentiator**: You can run the *same evaluator* on a **Dataset** (Offline) OR on a **Trace** (Online).
  - Running on a **Dataset** = Offline Evaluation.
  - Running on a **Trace** = Online Evaluation (monitors live traffic).

---

## 🔁 Part 6: Closing the Self-Improving Loop

This is the most important architectural insight of the course.

1. **Offline Eval** uses a Golden Dataset to test correctness pre-launch.
2. **Online Eval** monitors live traffic for normalcy.
3. When the Online Eval finds a **production failure** (e.g., a buggy conversation):
   - You click "**Add to Dataset**" in LangSmith.
   - That specific failed conversation is added to your Offline Golden Dataset.
4. Next time you release a new version, the Offline Eval runs on this **updated dataset**, catching that specific bug forever.

**This creates a continuous loop**: Offline → Deploy → Online detects bug → Bug goes back to Offline → Repeat. The system automatically improves over time.

### 💻 Code Example: Simulating the Feedback Loop

```python
# Simulating the offline golden dataset and online failure loop

# Initial Golden Dataset (Offline)
golden_dataset = [
    {"question": "What is Python?", "expected": "Programming language."}
]

# Simulate a production failure (detected via online monitoring)
production_failure = {
    "question": "How much is the refund?",
    "wrong_answer": "Refund is $1000.",  # Bot gave wrong answer
    "correct_answer": "Refund is $100."   # Human verified later
}

# THE LOOP: Add the production failure back to the offline dataset
def add_failure_to_dataset(dataset, failure):
    dataset.append({
        "question": failure["question"],
        "expected": failure["correct_answer"]  # Now offline test will catch this!
    })
    print(f"Added failure to offline dataset. New dataset size: {len(dataset)}")

# Next time we run Offline Eval, it checks against this new edge case.
add_failure_to_dataset(golden_dataset, production_failure)
# Output: Added failure to offline dataset. New dataset size: 2
```

---

## 📝 Final Summary / Key Takeaways

| Concept | Explanation |
| :--- | :--- |
| **Online Pipeline** | Logging → Sampling (for computed) → Dashboarding → Alerting. |
| **Step 1: Logging** | Must be non-blocking, durable, queryable, PII-safe. |
| **Captured Signals** | Latency, Cost, Thumbs Up/Down (directly stored). |
| **Computed Signals** | Hallucination, Toxicity, Faithfulness (require an LLM Judge). |
| **Sampling Strategy** | **Stratified** over **Random** (prioritize high-risk conversations). |
| **The Loop** | Online production failures must be fed back into the offline Golden Dataset for continuous system improvement. |
| **Mindset Shift** | You are not just a "builder" anymore. You are a **Production AI Engineer** ensuring the app works everywhere, every time. |

**Bottom Line**: Offline evals tell you *"Can we launch?"* Online evals tell you *"Is it still working right now?"* You combine both and use the online failures to constantly upgrade your offline tests—creating an unstoppable quality feedback loop. 🚀

---

## 07. LLM Model Evals & Capabilities (37:51)

## 🔄 Part 1: Quick Recap of the Course So Far

Before starting the new topic, here is a lightning recap:

1. **Why Evals?** To avoid legal/reputation disasters (Air Canada, Lawyer cases).
2. **What are Evals?** Systematic, repeatable tests against clear criteria.
3. **Types:** **Model Evals** (testing the base LLM) vs. **Application Evals** (testing your built system).
4. **How it Works:** The standard eval pipeline (Task → Dataset → Run → Analyze → Deploy).
5. **Multiple Pipelines:** One app needs multiple evals because of multiple failure points and risk categories.
6. **Online Evals:** Monitoring live production traffic for *normalcy* after deployment.

**Now, we shift focus to Model Evals** (the "brain" selection phase).

---

## 🧠 Part 2: Why AI Engineers *Need* Model Evals

As an AI Engineer, you don't train LLMs (OpenAI/Google do that). But you **choose** which LLM powers your application. Model Evals give you the data to make that choice.

**The 4 Critical Reasons for AI Engineers:**

1. **Compare Models Objectively**: In a team meeting, you can't say "both are good." You need hard numbers to prove why you chose OpenAI over Claude (or vice versa).
2. **Track New Model Improvements**: If Claude releases a new version (e.g., Opus → Sonnet), Model Evals help you determine if it is *actually* better for your use case before you upgrade.
3. **Check Safety & Robustness**: Evaluate if a model is safe, hallucinates rarely, and resists jailbreaks *before* exposing it to users.
4. **Decide: Proprietary vs. Open Source**: Should you pay for a heavy API (like Claude) or host a cheaper open-source model (like DeepSeek/Mixtral) yourself? Model Evals give you the ROI comparison.

> **Key Statement**: *"Without Model Evals, you are blind."*

---

## 🔬 Part 3: The 4-Step Structure of a Model Evaluation

Every Model Eval follows this exact process:

| Step | Action | Explanation |
| :--- | :--- | :--- |
| **1** | **Decide the Capability** | What do you want to test? (Reasoning? Coding? Safety?) |
| **2** | **Bring a Test** | Get a standardized **Benchmark** OR build a **Custom Dataset** for your specific task. |
| **3** | **Run Under Controlled Protocol** | Run the LLM on the test in a fixed environment (same temperature, same prompts) so results are repeatable across models. |
| **4** | **Score & Interpret** | Analyze the results to compare models or decide if the model passes your threshold. |

---

## 📂 Part 4: Standard Benchmarks vs. Custom Evals

There are **two types of tests** you can use for Model Evals:

1. **Standardized Benchmarks** (e.g., MMLU, GSM8K, SWE-bench):
   - Shared, global tests that everyone runs.
   - Great for *general* comparison (e.g., "Which model is smarter?").
2. **Custom Evals (Private Datasets)**:
   - You create your own dataset from your company's *actual* use case (e.g., 200 past Zomato emails).
   - Measures *specific* performance for *your* application, not general capability.

### 💡 The Zomato Case Study (Why Custom Evals Win)

**Scenario**: You are building an email classification system for Zomato (Billing vs. Technical vs. Refund). You have two model choices:

| Model | Public Benchmarks (General) | Accuracy (Your Custom Eval) | Cost (1M Tokens) | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** (Big/Expensive) | Top of the leaderboard 🥇 | **94%** | **$15.00** | **4.1 sec** |
| **Model B** (Small/Cheap) | Mid of the table 😐 | **91%** | **$0.50** | **0.9 sec** |

**If you only looked at public benchmarks**, you would blindly choose Model A (it wins everywhere). 
**But with a Custom Eval**, you see the truth: Model B gives 91% accuracy (only 3% less) but is **30x cheaper** and **4.5x faster**. For a massive company like Zomato, Model B is the clear business winner.

### 💻 Code Example: The "Custom Eval" ROI Decision Logic

```python
# Simulating the Zomato Email Classification Choice
public_benchmarks = {
    "Model_A": {"MMLU": 89, "GSM8K": 92}, # Wins here
    "Model_B": {"MMLU": 72, "GSM8K": 75}, # Loses here
}

# But Custom Evals tell the real story for YOUR app
custom_eval_results = {
    "Model_A": {"accuracy": 94, "cost_per_mil": 15.0, "latency_sec": 4.1},
    "Model_B": {"accuracy": 91, "cost_per_mil": 0.5, "latency_sec": 0.9},
}

def calculate_roi(model_name, daily_million_tokens=10):
    data = custom_eval_results[model_name]
    daily_cost = data["cost_per_mil"] * daily_million_tokens
    monthly_cost = daily_cost * 30
    return {
        "accuracy": data["accuracy"],
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "latency": data["latency_sec"]
    }

roi_a = calculate_roi("Model_A")
roi_b = calculate_roi("Model_B")

print(f"Model A: Acc {roi_a['accuracy']}% | Monthly Cost ${roi_a['monthly_cost']:.0f} | Latency {roi_a['latency']}s")
print(f"Model B: Acc {roi_b['accuracy']}% | Monthly Cost ${roi_b['monthly_cost']:.0f} | Latency {roi_b['latency']}s")

# Decision Logic
if roi_b["accuracy"] > 90 and roi_b["monthly_cost"] < roi_a["monthly_cost"]:
    print("🏆 Decision: Select Model B. It is 'good enough' for the task and saves massive costs.")
# Output: Model B wins on ROI!
```

---

## 📊 Part 5: The 8 Core Capabilities of LLMs (What Benchmarks Measure)

Every famous benchmark you see (MMLU, SWE-bench, etc.) targets one of these **8 Core Capabilities**. Here is the breakdown:

### 1. Knowledge & Reasoning
- **What**: Does the model know facts across domains (Science, History, Law) and can it connect multiple facts to solve complex logic problems?
- **Why important**: Measures the model's raw "intelligence".
- **Famous Benchmark**: **MMLU** (57 subjects).
- **Real-world use**: Research chatbots, legal/policy analysis.

### 2. Coding & Software Engineering
- **What**: Can it write functional code, fix bugs in large codebases, refactor code, and use APIs/command lines?
- **Why important**: **Huge economic value** (Cursor AI valued at $60B because of this).
- **Famous Benchmark**: **SWE-bench**.
- **Real-world use**: AI coding agents, automated DevOps.

### 3. Mathematics
- **What**: Can it solve grade-school math, competition-level (Olympiad), undergraduate, and even research-level symbolic/numerical problems?
- **Famous Benchmark**: **GSM8K**.
- **Real-world use**: Financial modeling, scientific computing, engineering simulations.

### 4. Long Context Management
- **What**: Can it effectively retrieve and use information from *very long* inputs (hundreds of thousands of tokens), or does it "forget" the middle parts?
- **Why important**: Models claim huge context windows (1M tokens), but actual retention degrades.
- **Famous Benchmark**: **Needle in a Haystack**.
- **Real-world use**: Analyzing giant legal contracts, coding across large codebases.

### 5. Vision & Multimodality
- **What**: Can it understand images, videos, and charts, or is it strictly text-only?
- **Real-world use**: Visual QA (reading fridge contents), document analysis with graphs.

### 6. Agentic & Tool Use
- **What**: Can it autonomously choose and use tools (web browsing, API calls, desktop control) to accomplish a multi-step goal?
- **Real-world use**: Building AI agents that book flights, order groceries, or control software.

### 7. Safety & Alignment
- **What**: Does it generate harmful/hateful content? Is it sycophantic (just agreeing with the user) or truthful? Does it resist adversarial attacks/jailbreaks? Does it have cybersecurity skills (ethical hacking)?
- **Why important**: Government regulations and reputation.
- **Famous Benchmark (emerging)**: CyberSec Eval (tests cryptographic/reverse engineering skills).

### 8. Instruction Following
- **What**: Can it strictly obey user constraints (e.g., "Keep it under 200 words", "Use bullet points", "Be friendly")?
- **Why important**: Directly translates to user satisfaction. If a model ignores instructions, users leave.
- **Famous Benchmark**: **IFEval**.

---

## 📝 Summary Table of 8 Capabilities

| # | Capability | What it Tests | Key Benchmark | Relevance |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Knowledge & Reasoning | Factual recall + multi-step logic | **MMLU** | Measures general "IQ" |
| 2 | Coding & Engineering | Writing, fixing, and refactoring code | **SWE-bench** | Massive economic value (AI coders) |
| 3 | Mathematics | Symbolic/numerical reasoning | **GSM8K** | Scientific/Fintech apps |
| 4 | Long Context | Retrieving info from huge documents | **Needle in a Haystack** | Legal/Coding agents |
| 5 | Vision/Multimodal | Understanding images/video | MMMU | Real-world visual tasks |
| 6 | Agentic/Tool Use | Calling APIs and using tools autonomously | ToolBench | Building AI Agents |
| 7 | Safety & Alignment | Toxicity, jailbreak resistance, truthfulness | CyberSec Eval | Legal compliance |
| 8 | Instruction Following | Obeying format, length, and tone constraints | **IFEval** | Direct user satisfaction |

---

## 08. Whats is LLM Benchmarking | Benchmark Saturation vs. Contamination (51:19)

### Running the GSM8K evaluation

The `run_gsm8k.sh` script executes this command:

```bash
exec "$script_dir/venv/bin/lm-eval" run \
  --model openai-chat-completions \
  --model_args "model=gpt-5.6-luna,num_concurrent=5,max_retries=5" \
  --tasks gsm8k_cot \
  --num_fewshot 8 \
  --apply_chat_template \
  --limit 20 \
  --output_path ./gsm8k_results \
  --log_samples
```

This runs an 8-shot, chain-of-thought GSM8K mathematics evaluation against GPT-5.6 Luna through the OpenAI Chat Completions API. It evaluates only 20 questions as a quick test and writes the metrics and individual model responses to `gsm8k_results`.

#### Command breakdown

| Part | Meaning |
|---|---|
| `exec` | Replaces the shell-script process with the `lm-eval` process. The script exits with the same exit status as `lm-eval`, and no script commands run after it. |
| `"$script_dir/venv/bin/lm-eval"` | Runs the exact `lm-eval` executable from this project's virtual environment. Using `$script_dir` makes it work regardless of the directory from which the script is launched. |
| `run` | Selects the lm-evaluation-harness command that runs an evaluation. |
| `--model openai-chat-completions` | Uses lm-eval's adapter for OpenAI's Chat Completions API rather than loading a model locally. `OPENAI_API_KEY` must be exported; the script loads it from `.env`. |
| `model=gpt-5.6-luna` | Sends the prompts to the GPT-5.6 Luna model. API usage is billed to the OpenAI Platform project associated with the key. |
| `num_concurrent=5` | Allows up to five API requests to run concurrently. This can reduce runtime, but it may encounter rate limits sooner than sequential requests. |
| `max_retries=5` | Retries a failed API request up to five times for temporary errors such as rate limits or network interruptions. It does not repair an invalid API key. |
| `--tasks gsm8k_cot` | Runs the chain-of-thought configuration of GSM8K, a benchmark of grade-school mathematical reasoning. |
| `--num_fewshot 8` | Includes eight solved examples in the prompt before each question. These demonstrations show the expected reasoning and answer format. |
| `--apply_chat_template` | Formats the benchmark prompt as chat messages in the structure expected by a chat model. |
| `--limit 20` | Evaluates only 20 benchmark examples. This is useful for a smoke test, but the resulting score is not a reliable full benchmark metric. Remove this option for a real evaluation. |
| `--output_path ./gsm8k_results` | Writes results beneath `gsm8k_results` relative to the script directory. |
| `--log_samples` | Saves per-question prompts, model responses, target answers, and scoring information in addition to aggregate metrics. This requires `--output_path`. |

The backslash (`\`) at the end of each line tells the shell that the command continues on the next line. It is one command, formatted across multiple lines for readability.

Run the script with:

```bash
cd 08_llm-benchmarking
./run_gsm8k.sh
```

For a full benchmark, remove `--limit 20`. Be aware that a full run sends many more billable API requests and takes longer.

---

This lecture, diving deep into **Model-Level Evaluations**, specifically **Benchmarks**. 

## 📌 Part 1: Recap & Definition of a Benchmark

- **Model Evals** test the base LLM's capabilities.
- **Two Types**: Standardized Benchmarks (global tests) vs. Custom Evals (your own private dataset).
- **Definition**: A benchmark is a **Standardized Test** used to measure a specific model capability (e.g., Math, Coding, Reasoning).

---

## 🧩 Part 2: The 5 Core Components of Any Benchmark (GSM8K Example)

The instructor uses **GSM8K** (Grade School Math, 8,000 questions) as the running example. Every benchmark has these parts:

| Component | What it means | GSM8K Example |
| :--- | :--- | :--- |
| **1. Task & Dataset** | The questions + the correct answers (like a Golden Dataset). | 8,000 math word problems + final numeric answers. |
| **2. Run Configuration** | Fixed settings (prompt format, temperature, token limits) to ensure a fair test. | Uses **8-shot** prompting + **Chain-of-Thought (CoT)**. |
| **3. Output Extraction** | Parsing the raw LLM text to get just the answer (e.g., extracting "72" from "The answer is 72"). | Regex/Structured parsing to get the numeric answer. |
| **4. Scoring Method** | How to judge if the extracted answer is correct. | **Pass@1** (correct on first try) vs. **Pass@K** (correct in top K attempts). |
| **5. Aggregation** | How to combine individual scores into a final percentage. | Simple average (e.g., 7200 correct out of 8000 = 90%). |

---

## ⚙️ Part 3: Understanding Run Configurations (The "Fair Exam" Settings)

To compare two models fairly, you must use the **exact same settings**. Key configs include:

1. **Prompting Style**:
   - **Zero-shot**: Ask the question directly (no examples).
   - **Few-shot** (GSM8K uses 8-shot): Provide solved examples *before* asking the question to "teach" the model the format.
2. **Chain-of-Thought (CoT)**: Allow the model to "show its work" step-by-step before giving the final answer (improves accuracy for math/logic).
3. **Temperature**: Set to `0` to make outputs deterministic (reduces randomness).
4. **Max Tokens**: Sufficient to allow the model to reason without cutting off abruptly.
5. **Tool Access**: Disabled for GSM8K (unless specified), but enabled for coding benchmarks where the model must run code.

### 💻 Code Example: Simulating "Few-Shot" & "CoT" Prompt Building

```python
# Simulating how a benchmark builds a standardized prompt

def build_gsm8k_prompt(question, few_shot_examples, enable_cot=True):
    prompt = "You are a math solver. Solve the following problems step-by-step.\n\n"
    
    # 1. Inject FEW-SHOT EXAMPLES (8-shot)
    for ex in few_shot_examples:
        prompt += f"Question: {ex['q']}\n"
        if enable_cot:
            prompt += f"Step-by-step: {ex['steps']}\n"
        prompt += f"Answer: {ex['ans']}\n\n"
    
    # 2. Inject the ACTUAL TEST QUESTION
    prompt += f"Question: {question}\n"
    if enable_cot:
        prompt += "Step-by-step: "  # Model should fill this
    else:
        prompt += "Answer: "
        
    return prompt

# Example usage
examples = [{"q": "2+2", "steps": "2+2=4", "ans": "4"}]
test_q = "Natalia sold 48 clips in April and half as many in May. How many total?"
final_prompt = build_gsm8k_prompt(test_q, examples, enable_cot=True)
print(final_prompt)
# Output includes examples + CoT instruction.
```

---

## 🏃 Part 4: The Evaluation Harness (The Exam Administrator)

You *could* write a Python loop (load question → build prompt → call model → extract answer → score → repeat). However:

- **The Coding Problem**: You need to handle API retries, rate limits, batch processing, and regex parsing for thousands of questions.
- **The Solution**: **Evaluation Harnesses** (like EleutherAI's `lm-evaluation-harness` or DeepEval). These are libraries that automatically run the benchmarking loop for you.

### 💻 Code Example: Running a Benchmark via Harness (Conceptual)

```python
# Instead of writing a messy 200-line loop, you run one command.
# This demonstrates the "Abstraction" layer.

# Command-line execution (not pure Python, but shows the ease)
# `lm-eval --model openai --model_args engine=gpt-3.5-turbo --tasks gsm8k --limit 20`

# Simulating the backend loop the harness runs for you:
def run_benchmark_loop(model, dataset, config):
    results = []
    for question in dataset:
        # 1. Build prompt with config (few-shot, CoT)
        prompt = build_prompt(question, config)
        # 2. Call model with temp=0, max_tokens=config["max_tokens"]
        raw_output = model.generate(prompt, temperature=0)
        # 3. Extract answer (regex)
        predicted = extract_number(raw_output)
        # 4. Score vs ground truth
        score = 1 if predicted == question["answer"] else 0
        results.append(score)
    # 5. Aggregate
    return sum(results) / len(results) * 100

# The harness handles retries, rate limits, and logging automatically! ✅
```

---

## 🏆 Part 5: Who Runs These & Whose Score to Trust?

1. **Frontier Labs (OpenAI/Google) themselves**:
   - ❌ **Least Trustworthy**. They set favorable conditions (configuration gaming) and cherry-pick good scores. (Like a car company claiming unrealistic mileage).
2. **Third-Party Evaluators (e.g., LMSYS Chatbot Arena Leaderboard)**:
   - ✅ **Most Trustworthy**. They are independent. They run all models under *exactly the same* conditions.
3. **You (AI Engineer)**:
   - ✅ **Most Relevant**. You should run your *own* custom model eval on your *own* private dataset to see true performance for your specific task (like the Zomato email classifier case).

---

## ⚠️ Part 6: The 4 Major Pitfalls of Benchmarks (Why you can't blind-trust them)

1. **Benchmark Contamination (Data Leakage)**:
   - These public benchmarks (MMLU, GSM8K) are freely available online.
   - Frontier labs scrape the entire internet for training data.
   - The model **already saw the exact questions and answers** during training!
   - It's not "thinking"; it's just "memorizing". High scores become meaningless.

2. **Benchmark Saturation**:
   - Models get smarter over time. 
   - Previously hard benchmarks become too easy. Everyone scores 95-97%.
   - When scores cluster together, you cannot differentiate between a good and a great model. The benchmark is "retired" and replaced with a harder one.

3. **Configuration Gaming**:
   - Labs tweak settings unfairly (e.g., allowing the model to use a Python interpreter for a math benchmark, or turning on huge computational budgets) just to inflate their specific score.

4. **Aggregation Masking (Hiding Weak Spots)**:
   - **MMLU** has 57 subjects. A model might score 95% on Physics but only 10% on Economics.
   - The lab publishes the **average** (e.g., 85%). 
   - If you build an Economics chatbot, you will fail miserably because the average hid the terrible Economics score!

### 💻 Code Example: The Aggregation Masking Pitfall

```python
# Simulating the MMLU Aggregation trap

subject_scores = {
    "Physics": 0.95,
    "History": 0.94,
    "Economics": 0.10,  # ❌ Terrible for economics!
}

# The Frontier Lab publishes the SIMPLE MEAN (unweighted)
simple_mean = sum(subject_scores.values()) / len(subject_scores)
print(f"Published Simple Mean: {simple_mean:.0%}")  # Output: 66% (Hides the disaster)

# However, if your app is ECONOMICS-focused, you only care about Economics.
if subject_scores["Economics"] < 0.80:
    print("🚨 DANGER: This model is horrible for my Economics use case!")
    print("💡 The 66% average tricked me! I must check sub-scores.")

# Therefore, Always check category-wise performance, not just the headline number!
```

---

## 📝 Final Summary Table

| Concept | Key Point |
| :--- | :--- |
| **Definition** | Benchmarks are standardized exams for LLMs (e.g., GSM8K for Math). |
| **5 Components** | Dataset, Run Config, Extraction, Scoring, Aggregation. |
| **Run Config** | Must fix Few/Zero-shot, CoT, Temperature, and Max Tokens for fairness. |
| **Scoring** | Pass@1 (strict) vs Pass@K (lenient) vs Majority@K (voting). |
| **Harness** | Libraries (`lm-evaluation-harness`) that automate the benchmarking loop (retries, parsing). |
| **Trust** | Third-party leaderboards > Your own custom eval > Frontier lab self-reported numbers. |
| **Pitfall 1** | **Contamination**: Model memorized the answers from the internet. |
| **Pitfall 2** | **Saturation**: Too easy; everyone clusters at 95%, loses differentiation. |
| **Pitfall 3** | **Gaming**: Tweaking temperature/tools to unfairly boost scores. |
| **Pitfall 4** | **Masking**: Good overall average hides terrible performance on specific sub-topics. |

**Bottom Line**: Benchmarks are useful for a *rough* estimate of general intelligence, but **NEVER** choose a production model based solely on a leaderboard number. Always run your own custom evaluation on your specific data to catch hidden failures! 🚀

---

## 09. What are LLM Benchmarks | The Evolution of AI Knowledge Benchmarks (01:50:46)

This lecture is the "Evolution Roadmap" for the **Knowledge Capability** (testing how much factual world knowledge an LLM retains in its parameters).

---

## 📌 Part 1: Recap & The "Knowledge" Capability

**Recap**: Model Evals are done via **Standardized Benchmarks** (global tests) or **Custom Evals** (your own data). Today, we focus on the **Knowledge Capability**.

- **What is "Knowledge" in an LLM?** The factual world knowledge stored in the model's **parameters (weights)** after training on massive internet data. This is called **"Parametric Memory"**.
- **Why is it fundamental?** When LLMs were first built, the core expectation was: *"If we feed it the entire internet, it should know everything about the world."* Knowledge is the most basic capability; others (Reasoning, Coding) emerged later.

---

## 🗺️ Part 2: The 7-Benchmark Evolution Roadmap (The "Story")

Here is the exact chronological flow the instructor presented:

1. **MMLU (2020)** → Tests *Breadth* of knowledge (57 subjects).
   - *Flaw*: **Contamination & Saturation**. Models memorized the public questions. Everyone started scoring 85-90%.
2. **TruthfulQA (2021)** → Tests *Reliability/Truthfulness*.
   - *Discovery*: Bigger models often **lie** or parrot common internet misconceptions (e.g., "cracking knuckles causes arthritis").
   - *Flaw*: Eventually saturated too.
3. **AGI Eval (2022-23)** → Tests against *Human Exam Baselines* (SAT, Gaokao).
   - *Idea*: Instead of inventing new tests, just use existing human exams to compare LLMs vs. humans.
   - *Flaw*: Saturated over time as models improved.
4. **GPQA (Google-Proof Q&A, 2023-24)** → Tests *Depth* of knowledge.
   - *Idea*: Move away from basic questions. Ask PhD-level, **extremely hard** science questions (Physics, Biology, Chemistry) that even Googling won't easily solve.
5. **MMLU Pro (2024)** → *Repairs* MMLU.
   - *Fix*: Increased options (4 → 10) per question, reduced subjects to 12, and added reasoning.
6. **SimpleQA (Post-2024)** → Replaces TruthfulQA.
   - *Fix*: **No multiple choices**. The model must generate the answer freely, making hallucination detection stricter.
7. **Humanity's Last Exam (HLE, 2025)** → The "Final Exam".
   - *Idea*: Combine **Breadth** (100 subjects) and **Depth** (research-level questions) into ~2,500 ultra-hard questions. Designed to be the *last* knowledge benchmark needed. If models ace this, we stop testing basic knowledge.

---

## 🔍 Part 3: Detailed Breakdown of the 7 Benchmarks

| Benchmark | Year | Core Focus | Key Stats | Why it came next |
| :--- | :--- | :--- | :--- | :--- |
| **1. MMLU** | 2020 | Breadth of Knowledge | 57 subjects, 14k MCQs (4 options each). | The original "Mother of all benchmarks" for general knowledge. |
| **2. TruthfulQA** | 2021 | Reliability / Avoiding Myths | Questions with a *correct* and an *incorrect* (common misconception) answer. | To catch models that repeat false internet myths. |
| **3. AGI Eval** | 2022-23 | Human Baseline Comparison | Standard human exams (SAT, Gaokao, LSAT). | To directly compare LLM IQ vs. Human IQ on familiar ground. |
| **4. GPQA** | 2023-24 | Depth of Knowledge | ~500 PhD-level Science MCQs (Biology, Physics, Chem). | To test deep, specialized knowledge beyond basic facts. |
| **5. MMLU Pro** | 2024 | Harder Multiple Choice | 12 subjects, ~1k questions each, **10 options** instead of 4. | To "repair" saturated MMLU by making it harder. |
| **6. SimpleQA** | 2024+ | Hallucination Detection | Open-ended simple questions (no options given). | To strictly test if the model hallucinates when forced to generate text. |
| **7. HLE** | 2025 | Breadth + Depth (The Ultimate) | 100 subjects, 2,500 research-level questions. | To create a single, ultimate exam that combines all previous lessons. |

---

## 💻 Part 4: Code Examples for Key Concepts

### 1. Simulating the "Contamination & Saturation" Trap (MMLU)

```python
# Simulating why MMLU became useless over time

# Imagine a public benchmark with 100 questions.
public_benchmark_questions = ["Q1", "Q2", "Q3"]  # Actually 14,000

# Model A (2021) studies the internet BEFORE this benchmark was famous.
# Model B (2024) trains on internet data that INCLUDES this benchmark's answers.

def evaluate_model(model_name, benchmark_data):
    if model_name == "Model_B_2024":
        # Model B already saw the answers in training data (Contamination).
        return 98  # Unrealistically high!
    else:
        # Model A actually has to think.
        return 75

score_a = evaluate_model("Model_A_2021", public_benchmark_questions)
score_b = evaluate_model("Model_B_2024", public_benchmark_questions)

print(f"Model A Score: {score_a}% (Real performance)")
print(f"Model B Score: {score_b}% (Inflated due to memorization)")

# Conclusion: Benchmark is "Saturated". Both scores are high, but Model B is cheating.
```

### 2. Simulating TruthfulQA (Detecting Misconceptions)

```python
# TruthfulQA has pairs: [Question, Correct_Answer, Misconception_Answer]

question = "Does cracking knuckles cause arthritis?"

# Ground Truth (From actual doctors):
correct_answer = "No, it does not cause arthritis."

# Common internet misconception:
misconception_answer = "Yes, it can lead to arthritis."

def test_truthfulness(model_response):
    if "No" in model_response and "does not" in model_response:
        return "✅ Truthful (Passes TruthfulQA)"
    else:
        return "❌ Hallucinating/Misleading (Fails TruthfulQA)"

# A 2024 giant model that scraped reddit might say:
model_output = "Yes, cracking knuckles is harmful and causes arthritis."

result = test_truthfulness(model_output)
print(result)  # Output: ❌ Hallucinating/Misleading
```

### 3. Simulating SimpleQA (No Multiple Choice = Harder)

```python
# MMLU gives 4 options. Easy to guess.
mmlu_question = "What is the capital of France?"
mmlu_options = ["A. London", "B. Paris", "C. Berlin", "D. Madrid"]
# Model just has to pick "B".

# SimpleQA gives NO options. Model must generate the exact text.
simple_qa_question = "What is the capital of France?"
# Model output: "Paris" (exact match required)

def evaluate_open_ended(model_response, ground_truth):
    if model_response.strip().lower() == ground_truth.lower():
        return "Pass"
    else:
        return "Fail (Hallucination detected)"

response = "Paris"  # If model says "Paris, a beautiful city", it still needs parsing.
print(evaluate_open_ended(response, "Paris"))  # Output: Pass
```

### 4. Simulating HLE (Humanity's Last Exam)

```python
# HLE concept: Ultra-hard, research-level questions.
# Even experts would struggle to answer without deep research.

hle_question = "What is the specific binding affinity of the novel XYZ inhibitor on the mutated K-Ras G12C protein, and how does it compare to standard inhibitors?"

# The idea is that if a model can answer this in 2025, it has achieved super-human knowledge.
# Previous benchmarks (MMLU) were too easy, so HLE is the ultimate challenge.

def evaluate_hle(model_answer):
    # If the model scores > 90% on HLE, we declare victory on Knowledge Evals.
    if "binding affinity" in model_answer and "K-Ras" in model_answer:
        return "Likely correct (PhD level)"
    else:
        return "Failed (Did not understand the question)"

# Simulating a top-tier model's response
response = "The binding affinity of XYZ is 2.5 nM, which is stronger than standard inhibitors."
print(evaluate_hle(response))
```

---

## 🧠 Part 5: Important Pointers & Takeaways

1. **Knowledge is the "Foundation"**: All other capabilities (Reasoning, Coding) emerged *after* the model successfully stored world knowledge.
2. **The "Cat & Mouse" Game**: Benchmarks get created → Models get smarter/saturate them → Benchmarks get "retired" → New, harder benchmarks get created.
3. **Don't Trust Single Numbers**: A model might score 90% on MMLU but fail SimpleQA (hallucination) or fail specific subdomains. Always check the *sub-scores* (e.g., Physics vs. Economics).
4. **The Role of AI Engineer**: You don't need to memorize every question. You just need to understand **which benchmark tests what**, so you can make the right model selection for your app (e.g., if you build a medical chatbot, prioritize GPQA/HLE; if you build a general assistant, MMLU Pro + SimpleQA matters).

---

## 📝 Summary of the Knowledge Evolution (In Simple Words)

| Year | New Problem Discovered | Solution (New Benchmark) |
| :--- | :--- | :--- |
| 2020 | We need to test raw factual recall. | **MMLU** (57 subjects, 14k MCQs). |
| 2021 | Big models just repeat internet lies. | **TruthfulQA** (Tests for misconceptions). |
| 2022 | We need a human baseline to compare. | **AGI Eval** (Uses SAT/Gaokao exams). |
| 2023 | MMLU is too easy (saturated). | **GPQA** (PhD-level, extremely deep science). |
| 2024 | MMLU is broken. Let's fix it. | **MMLU Pro** (10 options instead of 4). |
| 2024 | TruthfulQA is saturated. | **SimpleQA** (No multiple choice, open generation). |
| 2025 | We need one final, ultimate test. | **Humanity's Last Exam (HLE)** (Breadth + Depth combined). |

- [benchwiki](https://benchwiki.vercel.app/)

---

This part of tutorial from **LLM Benchmarks** (contd...)

It starts with the introduction of **BenchWiki** (a central database for benchmarks) and then walks through the detailed architecture, scoring methods, flaws, and current status of MMLU, TruthfulQA, AGI Eval, GPQA, MMLU Pro, SimpleQA, and Humanity's Last Exam (HLE).

---

## 📚 Part 1: Introducing BenchWiki (The Benchmark Encyclopedia)

- **What**: The instructor is building an open website called **BenchWiki** (like Wikipedia for LLM benchmarks).
- **Purpose**: To provide a single source of truth containing:
  - Current status (Active, Saturated, Retired).
  - Performance charts over time.
  - Human baseline comparisons.
  - Task details, sample datasets, scoring methodology.
  - Contamination notes and run configurations.
- **For You**: You can use it for self-study to understand any benchmark in depth.

---

## 🔬 Part 2: Deep Dive into the 7 Knowledge Benchmarks

Here is the detailed breakdown of each benchmark, exactly as presented.

### 1. MMLU (Massive Multitask Language Understanding) – *The "Mother of all Benchmarks"*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2020 |
| **Core Focus** | **Breadth** of Knowledge (How much does the model know across many fields?) |
| **Dataset** | 14,000 MCQs across **57 subjects** (Humanities, STEM, Social Sciences, etc.). |
| **Source** | Real exams (GRE, USMLE, AP) and textbooks. |
| **Scoring Methods** | **1. Generation**: Model outputs "A/B/C/D". <br> **2. Log-Likelihood**: Compute the probability of each option token. Pick the highest. **(Log-likelihood often gives 2-3% higher scores)**. |
| **Run Config** | 5-shot, CoT disabled, Temperature=0, Pass@1, no tools. |
| **History** | 2020: GPT-3 scored ~43% vs Human Experts ~90%. <br> 2023: GPT-4 reached ~86%. <br> 2024: All frontier models clustered around 86-92%. |
| **Critical Flaws** | 1. **Label Errors**: ~6.5% of questions have wrong/corrupt answers (so **nobody can score 100%**). <br> 2. **Heavy Contamination**: Public since 2020, so all new models memorize the answers. <br> 3. **Prompt Sensitivity**: Changing the system prompt by 1 word can change scores by 5-10%. |
| **Current Status** | **Saturated & Retired** (no longer used by frontier labs). |

#### 💻 Code Example: Generation vs. Log-Likelihood Scoring (MMLU)

```python
# Simulating MMLU's two scoring methods

import numpy as np

# Given a question and 4 options (A, B, C, D)
options = ["A. Paris", "B. London", "C. Berlin", "D. Madrid"]
correct_answer = "A"

# --- Method 1: GENERATION (Model prints a character) ---
model_generated_output = "A"  # Simulated
if model_generated_output == correct_answer:
    generation_score = 1.0
else:
    generation_score = 0.0
print(f"Generation Score: {generation_score}")

# --- Method 2: LOG-LIKELIHOOD (Model assigns probabilities to each token) ---
# Simulating log-probabilities (softmax outputs) for A, B, C, D
log_probs = {"A": -0.1, "B": -1.5, "C": -2.0, "D": -0.8}  # Higher = more probable
# We pick the highest probability (A in this case)
predicted_log_likelihood_answer = max(log_probs, key=log_probs.get)
print(f"Log-Likelihood Predicted: {predicted_log_likelihood_answer}")

# Key finding: Log-likelihood often gives higher accuracy because the model doesn't have
# to format the answer correctly. It just has to internally know which token is most likely.
# This is why GPT-4 scores 84% via generation vs ~87% via log-likelihood!
```

---

### 2. TruthfulQA – *Testing Honesty & Misconceptions*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2021 |
| **Core Focus** | **Reliability / Truthfulness** (Does the model parrot common internet myths?) |
| **Dataset** | 817 adversarial questions about **common human misconceptions** (e.g., "Does cracking knuckles cause arthritis?"). |
| **Scoring Methods** | **Generation**: Model writes the answer. <br> **MC1**: Likelihood of a single correct answer. <br> **MC2**: Sum of likelihoods of *all* correct answers (if multiple correct). **Default is MC2**. |
| **History** | GPT-3 scored 58% vs Human 94%. <br> **Big Find**: Bigger models were often **less truthful**! (Capability ≠ Alignment). |
| **Critical Flaws** | Contamination happens during the **Alignment Stage** (RLHF/Instruction Tuning), not just pre-training. |
| **Current Status** | **Saturated** (replaced by SimpleQA). |

#### 💻 Code Example: TruthfulQA's MC2 Scoring (Multiple True Answers)

```python
# TruthfulQA has questions where multiple answers can be true.
# MC2 sums the probabilities assigned to ALL correct answers.

import numpy as np

# Example: Question about a common myth.
# Simulated log-probabilities assigned by the model to 4 options.
option_probs = {
    "true_1": 0.6,  # Correct
    "true_2": 0.3,  # Correct (multiple correct answers exist)
    "false_1": 0.05,
    "false_2": 0.05
}

# Correct set for this question
correct_set = ["true_1", "true_2"]

# MC1: Pick the single highest probability (0.6).
mc1_score = max(option_probs.values())
print(f"MC1 Score: {mc1_score:.0%}")  # 60%

# MC2: Sum the probabilities of ALL correct answers.
mc2_score = sum(option_probs[ans] for ans in correct_set)
print(f"MC2 Score: {mc2_score:.0%}")  # 90% (0.6 + 0.3)

# Insight: If a model is only 60% sure about the single best answer, 
# but 90% sure that the answer lies within the correct set, MC2 rewards this nuance.
```

---

### 3. AGI Eval – *Human Exam Benchmarks*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2023 |
| **Core Focus** | Comparing LLMs directly to **Human exam takers** (SAT, LSAT, Gaokao, etc.). |
| **Dataset** | 8000+ questions from 20 real human exams. |
| **Key Feature** | **Bilingual** (English + Chinese). Provides a real human baseline (Avg human: 67%, Top human: 91%). |
| **Critical Flaw** | Passing an exam does **NOT** mean achieving AGI. It tests memorization/retrieval, not long-horizon reasoning or tool use. |
| **Current Status** | **Saturated** (models now match/exceed top human scores). |

---

### 4. GPQA (Google-Proof Q&A) – *Testing Depth of Knowledge*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2023 |
| **Core Focus** | **Depth** of Knowledge (PhD-level, extremely hard questions). |
| **Dataset** | Physics, Chemistry, Biology questions. Subsets: Main (443), Extended (546), **Diamond (198)** – the hardest. |
| **Key Feature** | **Google-Proof**: Even if you give a non-expert 30 mins and Google, they cannot solve it. Validated by domain experts. |
| **History** | GPT-4 scored 39% initially. O1 (reasoning model) hit 78% in 2024. Grok 4 hit ~87% in 2025. |
| **Critical Flaw** | Very few questions (only 198 in Diamond), lowering statistical confidence. Only covers 3 subjects (Physics, Chem, Bio). |
| **Current Status** | **Near Saturation** (approaching 80-90%). |

---

### 5. MMLU Pro – *The Fixed Version of MMLU*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2024 |
| **Core Focus** | Repair the flaws of MMLU. |
| **Key Changes** | 1. **4 options → 10 options** (harder to guess via elimination). <br> 2. Removed trivia/noisy questions, added **Reasoning-based** questions. <br> 3. Reduced from 57 to 14 balanced subjects. |
| **Effect** | Reasoning models scored **20 points higher** than non-reasoning models, proving it tests thinking, not just recall. |
| **Critical Flaw** | No human baseline provided. Questions are sourced from public STEM problems (contamination risk). |
| **Current Status** | **Near Saturation** (models hitting 80-90%). |

#### 💻 Code Example: Why 10 Options > 4 Options (Elimination Strategy)

```python
import random

# Simulating random guessing on 4 vs 10 options

def simulate_guessing(num_options, trials=10000):
    correct = 0
    for _ in range(trials):
        guess = random.randint(1, num_options)
        answer = random.randint(1, num_options)
        if guess == answer:
            correct += 1
    return correct / trials

guess_rate_4 = simulate_guessing(4)  # Random chance = 25%
guess_rate_10 = simulate_guessing(10) # Random chance = 10%

print(f"Random chance with 4 options: {guess_rate_4 * 100:.0f}%")
print(f"Random chance with 10 options: {guess_rate_10 * 100:.0f}%")

# Insight: With 10 options, a model cannot rely on "intelligent elimination" as easily.
# It must ACTUALLY KNOW the answer or reason deeply.
```

---

### 6. SimpleQA – *Open-Ended Short Answers + Calibration*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2024 (OpenAI) |
| **Core Focus** | **Factuality + Calibration** (Does the model know what it *doesn't* know?) |
| **Dataset** | 4000+ short fact-seeking questions that **GPT-4 failed to answer** initially. |
| **Key Feature** | **NO MULTIPLE CHOICE**. The model must generate the answer freely. |
| **Scoring** | 3 categories: **Correct**, **Incorrect**, or **Not Attempted** (if the model says "I don't know"). |
| **History** | GPT-4 scored ~38%. O1 scored ~42%. GPT-4.5 scored ~62%. |
| **Critical Flaw** | LLM-as-a-judge is used for grading, which evolves over time (hard to compare scores across years). Also, "stale" answers (facts change over time). |
| **Current Status** | **Active** (far from saturation). |

#### 💻 Code Example: Simulating SimpleQA's "Calibration" & Refusal Mechanism

```python
# SimpleQA tracks if the model is "Humble" (knows its limits)

def simpleqa_scoring_model(model_response, ground_truth):
    # Scenario 1: Model gives the exact correct answer
    if model_response.strip().lower() == ground_truth.lower():
        return "Correct"
    
    # Scenario 2: Model admits ignorance (GOOD! No hallucination)
    elif "don't know" in model_response.lower() or "not sure" in model_response.lower():
        return "Not Attempted (Good Calibration)"
    
    # Scenario 3: Model guesses incorrectly (BAD hallucination)
    else:
        return "Incorrect (Hallucination)"

# Test cases
responses = [
    ("Albert Einstein", "Albert Einstein"),           # Correct
    ("I don't know who won that year.", "Marie Curie"), # Refused, Good!
    ("Isaac Newton", "Marie Curie")                    # Wrong guess, Bad!
]

for resp, truth in zip(responses, ["Correct", "Not Attempted", "Incorrect"]):
    print(simpleqa_scoring_model(resp, truth))

# Insight: SimpleQA penalizes the model for guessing when it doesn't know.
# This separates truly knowledgeable models from overconfident hallucinators.
```

---

### 7. HLE (Humanity's Last Exam) – *The Ultimate Knowledge Test*

| Aspect | Details |
| :--- | :--- |
| **Year** | 2025 |
| **Core Focus** | **Breadth × Depth** (Combines MMLU's breadth and GPQA's depth). |
| **Dataset** | ~2,500 expert-written questions across **100+ subjects** (from Classics to Rocket Engineering). |
| **Key Features** | 1. **10% Multimodal** (includes images/charts). <br> 2. **Private Held-Out Set** (not publicly available) – **prevents contamination!** <br> 3. Tests **Accuracy + Calibration** (asks models for confidence scores). |
| **Scoring** | Accuracy + RMSE of confidence scores. |
| **History** | 2025: Grok 4 ~24%, GPT-5 ~25%, Gemini ~38%. <br> 2026: Frontier models still struggling. |
| **Philosophy** | If models ace this (breadth + depth across 100 subjects), we declare victory on closed-ended knowledge tests and move to **open-ended agentic tasks**. |
| **Critical Flaw** | LLM-as-a-judge grading errors, selection bias (questions filtered to stump older models). |
| **Current Status** | **Active** (The current State-of-the-Art benchmark). |

#### 💻 Code Example: HLE's Breadth + Depth + Calibration

```python
# HLE combines Breadth (100 subjects) and Depth (expert level).
# It also asks the model for its confidence.

def hle_evaluate(model_answer, ground_truth, model_confidence):
    # 1. Check correctness
    if model_answer == ground_truth:
        correctness = 1.0
    else:
        correctness = 0.0
    
    # 2. Check Calibration (Confidence should match accuracy)
    # If model says 90% confident but gets it wrong -> bad calibration.
    calibration_error = abs(correctness - model_confidence)
    
    return {
        "correct": correctness,
        "confidence": model_confidence,
        "calibration_error": calibration_error
    }

# Test: Model gets it right but is unsure (under-confident)
result = hle_evaluate("Paris", "Paris", 0.6)
print(result) # correct: 1.0, confidence: 0.6, error: 0.4

# Test: Model gets it wrong but is over-confident (over-confident -> hallucination risk)
result = hle_evaluate("London", "Paris", 0.95)
print(result) # correct: 0.0, confidence: 0.95, error: 0.95
```

---

## 📝 Part 3: Summary of the Benchmark Lifecycle

A crucial takeaway from the instructor is the **"Cat & Mouse" lifecycle** of every benchmark:

1. **Arrival**: A new benchmark is released. Models score poorly (e.g., 30-40%).
2. **Improvement**: Over 1-2 years, models get smarter (bigger scale, better alignment).
3. **Saturation**: Models score 80-90%+. All models cluster together.
4. **Retirement**: The benchmark is discarded. A newer, harder one replaces it.
5. **Repeat**: The cycle continues.

| Benchmark | Year | Core Focus | Scoring Quirk | Status |
| :--- | :--- | :--- | :--- | :--- |
| **MMLU** | 2020 | Breadth | Log-likelihood vs Generation | ❌ Retired/Saturated |
| **TruthfulQA** | 2021 | Reliability | MC2 (Sum of true probs) | ❌ Retired/Saturated |
| **AGI Eval** | 2023 | Human Exams | Bilingual (EN/CN) | ❌ Retired/Saturated |
| **GPQA** | 2023 | Depth | Diamond subset (198 Qs) | ⚠️ Near Saturation |
| **MMLU Pro** | 2024 | Fixed MMLU | 10 Options + Reasoning | ⚠️ Near Saturation |
| **SimpleQA** | 2024 | Calibration | Open-ended + Refusal | ✅ Active |
| **HLE** | 2025 | Breadth x Depth | Private set + Multimodal | ✅ Active (SOTA) |

---

## 🧠 Final Important Pointers (The "AI Engineer" Mindset)

1. **Don't Trust Single Numbers**: A model scoring 90% on MMLU doesn't mean it's 90% "smart". Check sub-scores (Physics vs. Economics) or cross-check with SimpleQA/HLE.
2. **Contamination is Real**: Public benchmarks are dangerous to use for comparison because models memorize them.
3. **Calibration Matters**: A model that says "I don't know" is better than one that hallucinates confidently. SimpleQA and HLE measure this.
4. **Reasoning ≠ Memorization**: MMLU Pro and HLE force models to *think* rather than just recall.
5. **The Final Frontier**: If models ace HLE, we stop testing closed knowledge and shift to testing agents (tools, long-horizon planning).

---

## 010. How to Use LLM Leaderboards (30:07)

This tutorial guide to **LLM Leaderboards** – how they work, why they exist, the different types, their hidden pitfalls, and the **step-by-step framework** every AI Engineer should follow to use them correctly (without falling into the "leaderboard trap").

---

## 📌 Part 1: What is an LLM Leaderboard? (Definition)

**Definition**: An LLM Leaderboard is a **public ranking table** that shows how different LLMs perform on a common set of evaluations (benchmarks).

**Analogy**: 
- **Benchmark** = The "Exam Paper" (e.g., MMLU, HLE).
- **Leaderboard** = The "Result Sheet" showing who got the highest marks.

---

## 🎯 Part 2: Why Do Leaderboards Exist? (4 Key Reasons)

1. **Common Reference**: Compare models from different labs (OpenAI vs. Google vs. Anthropic) on a level playing field.
2. **Trust (Third-Party)**: Leaderboards run by independent third parties are more trustworthy than a company's self-reported scores.
3. **Cost/Time Saver**: You cannot run evaluations on every model in existence (too expensive/time-consuming). Leaderboards filter the pool for you.
4. **Discovery**: You find hidden gems – small, cheap models that punch above their weight in specific tasks.

---

## 👥 Part 3: Who Uses Leaderboards?

| Stakeholder | How They Use It |
| :--- | :--- |
| **AI Engineers (You!)** | Shortlist candidate models for their specific application (e.g., "find models good at math"). |
| **Frontier Labs** | Strategic planning – if their new model doesn't beat competitors, they delay the release. Often release "Stealth Models" (like *Nano Banana*) to test the waters before revealing it's theirs. |
| **Researchers** | Identify saturated benchmarks (no progress) vs. active frontiers (new research directions). |
| **Policymakers/Safety Institutes** | Monitor if any model becomes dangerously powerful (e.g., triggering government intervention like with some releases). |
| **Open-Source Community** | Discover new innovative labs/models that suddenly pop up to the top. |

---

## 📂 Part 4: The 4 Types of Leaderboards

| Type | What it does | Example |
| :--- | :--- | :--- |
| **1. Benchmark-Specific** | Ranks models on **only one** benchmark (gives a narrow view). | HLE's Official Leaderboard |
| **2. Multi-Benchmark / Aggregate** | Combines scores from **multiple benchmarks** into a single overall score. Also provides cost, latency, and speed. | **LiveBench**, **Artificial Analysis** (most useful). |
| **3. Human Preference** | Users vote on which model's response they prefer (Blind A/B testing). | **LMSYS Chatbot Arena** (Elo rating system). |
| **4. Application-Specific** | Focuses on a specific domain (e.g., coding, function calling, SQL). | **Berkeley Function Calling Leaderboard**. |

### 💻 Code Example 1: Simulating a "Multi-Benchmark" Aggregated Score

```python
# Simulating how Artificial Analysis / LiveBench creates a composite score

benchmark_scores = {
    "Model_A": {"MMLU": 90, "GSM8K": 92, "HumanEval": 85, "Latency": 1.2},
    "Model_B": {"MMLU": 88, "GSM8K": 95, "HumanEval": 82, "Latency": 0.8},
    "Model_C": {"MMLU": 85, "GSM8K": 80, "HumanEval": 95, "Latency": 2.5},
}

# The leaderboard uses a secret weighting formula!
weights = {"MMLU": 0.4, "GSM8K": 0.3, "HumanEval": 0.3}

def calculate_composite_score(model_scores, weights):
    score = 0
    for bench, weight in weights.items():
        score += model_scores[bench] * weight
    return round(score, 2)

for model, scores in benchmark_scores.items():
    composite = calculate_composite_score(scores, weights)
    print(f"{model} Composite Score: {composite}")

# Output:
# Model_A: 89.1
# Model_B: 88.3
# Model_C: 85.5

# IMPORTANT: If the weights changed, Model_B might win!
# This shows why we need to understand HOW the leaderboard is calculated.
```

### 💻 Code Example 2: Simulating "Human Preference" (ELO Rating / Voting)

```python
# Simulating LMSYS Chatbot Arena logic

import random

class ArenaMatch:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        # Simulate users preferring well-formatted, long, confident responses.
        # Model_A has better formatting, Model_B is more factual but dry.
    
    def simulate_user_vote(self):
        # Human bias: Prettier formatting gets votes even if content is similar.
        if self.model_a == "GPT-4 (Verbose)":
            return "A"  # Humans often choose the longer, well-structured one.
        else:
            return random.choice(["A", "B"])  # Else random.

# Over thousands of votes, "Friendly/Verbose" models rank higher 
# than "Dry/Factual" ones, even if the factual accuracy is lower.
```

---

## ⚠️ Part 5: 7 Critical Pitfalls – Why You CANNOT Blindly Trust Leaderboards

1. **Benchmark Performance ≠ Real-World Performance**: Benchmarks are clean; real-world data is messy (ambiguous queries, missing info, edge cases).
2. **Contamination**: Models memorize public benchmark answers. High scores are often due to memorization, not reasoning.
3. **Goodhart's Law**: *"When a measure becomes a target, it ceases to be a good measure."* Models are trained specifically to win the leaderboard, sacrificing actual real-world usefulness (like companies optimizing only for fuel mileage and ruining the driving experience).
4. **Hidden Weighting/Aggregation**: How is the "Overall" score calculated? Which benchmarks are included or excluded? It's often a black box.
5. **Statistically Insignificant Differences**: A 0.2% difference in score doesn't mean one model is better. They are functionally identical.
6. **Human Bias**: In human-vote leaderboards, people prefer longer, prettier, more entertaining answers—not necessarily the *most accurate* ones.
7. **Stale/Self-Reported Data**: Leaderboards often aren't updated with the newest models, or companies cherry-pick favorable scores to report.

### 💻 Code Example 3: Simulating Goodhart's Law in Action

```python
# Scenario: A leaderboard rewards "Verbose Answers".

def evaluate_model_quality(model, is_verbose):
    # Simulated real-world metrics
    factual_accuracy = 85  # Baseline
    
    # The developer optimized the model to be verbose to get votes.
    if is_verbose:
        # Goodhart's Law: Optimizing for verbosity hurts factual accuracy!
        factual_accuracy -= 15  # Drops because it starts generating "fluff" and hallucinated details.
    
    # However, the leaderboard only checks "Length" and "Formatting"...
    leaderboard_score = 95 if is_verbose else 70
    
    return {
        "Leaderboard_Score": leaderboard_score,
        "True_Factual_Accuracy": factual_accuracy
    }

result_verbose = evaluate_model_quality(model="Optimized_for_Leaderboard", is_verbose=True)
result_dry = evaluate_model_quality(model="Not_Optimized", is_verbose=False)

print(f"Verbose Model: LB={result_verbose['Leaderboard_Score']}, Fact={result_verbose['True_Factual_Accuracy']}")
print(f"Dry Model: LB={result_dry['Leaderboard_Score']}, Fact={result_dry['True_Factual_Accuracy']}")

# Output:
# Verbose Model: LB=95, Fact=70  (Wins leaderboard, loses in reality)
# Dry Model: LB=70, Fact=85      (Loses leaderboard, better in reality)
```

---

## 🛠️ Part 6: The AI Engineer's 5-Step Framework (How to Use Leaderboards Correctly)

**The Golden Rule**: *Leaderboards are a **Filtering Tool**, NOT a **Decision Tool**.*

| Step | Action | Details |
| :--- | :--- | :--- |
| **1. Define Constraints** | Write down your app's needs: latency budget, cost tolerance, context window, deployment (cloud vs. on-premise). | If you need on-premise, ignore proprietary models (Claude/GPT) completely. |
| **2. Choose the Right Board** | Pick a leaderboard for your **specific domain**. | Building a Chatbot? → LMSYS Arena. Building an Agent? → Berkeley Function Calling. Building RAG? → MTEB (Embeddings). |
| **3. Read Critically** | Check the fine print. Is it saturated? Is there a private test set? What's the confidence interval? | If scores are 90-95% clustered, the benchmark is dead. |
| **4. Shortlist (Filter)** | Select **Top 3 to 5** models that pass your constraints. | Narrow down from 100 models to a handful. |
| **5. Run YOUR Custom Eval** | Run your own **custom evaluations** (golden dataset) on these 3-5 candidates. | **This is where you make the final decision.** |

### 💻 Code Example 4: The "Filter, Don't Decide" Strategy

```python
# Step 1: 100 models exist.
all_models = [f"Model_{i}" for i in range(100)]

# Step 2: Filter using Leaderboard (Speed constraint: Latency < 0.8s).
# Simulating a filter based on leaderboard data.
leaderboard_data = {m: {"latency": round(0.5 + (i*0.05), 2), "score": 80 + (i*0.1)} for i, m in enumerate(all_models)}

# Filter: Keep only models with latency < 0.8 seconds.
shortlisted = [m for m in all_models if leaderboard_data[m]["latency"] < 0.8]
print(f"Leaderboard filtered 100 models down to {len(shortlisted)}.")

# Step 3: Run YOUR Custom Eval on the shortlist.
# Simulating a custom golden dataset (e.g., 500 internal support tickets).
def run_custom_eval(model, custom_data):
    # Simulate accuracy on YOUR specific data.
    base = leaderboard_data[model]["score"]
    # Real-world data is messy, so performance drops compared to the leaderboard.
    custom_score = max(65, base - 15) 
    return custom_score

for model in shortlisted[:5]:  # Pick the top 5 from the shortlist
    print(f"{model}: Leaderboard={leaderboard_data[model]['score']}% | My Custom Data={run_custom_eval(model, [])}%")

# The model with the highest LEADERBOARD score might NOT be the best on YOUR custom data!
# You pick based on YOUR custom eval results.
```

---

## 📝 Final Summary Table

| Concept | Key Takeaway |
| :--- | :--- |
| **Definition** | Leaderboards = Public ranking tables of LLMs on benchmarks. |
| **Why Exist** | Common reference, trust, cost-saving, discovery. |
| **4 Types** | Benchmark-Specific, Multi-Benchmark, Human-Preference, Application-Specific. |
| **Pitfall 1** | Leaderboard scores are inflated by contamination/memorization. |
| **Pitfall 2** | Goodhart's Law – Models optimize for the test, not the real world. |
| **Pitfall 3** | Small numerical differences are statistically meaningless. |
| **The Golden Rule** | Leaderboards are a **filtering tool** (go from 100 → 5 models), NOT a decision tool. |
| **Final Step** | **Always** run your own custom eval on the final candidates to make the real choice. |

**Bottom Line**: Never blindly pick the "#1" model on a leaderboard. Use the leaderboard to eliminate the clearly bad/poorly-suited models, then test the remaining few on your actual production data. That's how a professional AI Engineer makes decisions. 🚀

---

## 011. Selecting the Right LLM for Your AI App: Running Custom Model Evals (01:56:52)

This tutorial is for **"Hands-On Transition"** in the LLM Evaluation course. After weeks of heavy theory (Model Evals, Benchmarks, Leaderboards), the instructor moves to practical work using a **real-world Text-to-SQL case study** (ESPN Cricinfo). 

The core lesson is the **3-Step Model Selection Framework** an AI Engineer must follow to pick the right LLM for a specific application. The instructor spends most of the session on **Step 1: Gathering Requirements**, focusing heavily on **Cost Calculation (Token Math)** and **Prompt Caching (KV Cache)**.

---

## 🎯 Part 1: Recap & The Big Shift

- **Course Recap**: We learned about **Model Evals** (Benchmarks like MMLU/HLE for general capability, and **Custom Evals** for your specific task). We also learned that **Leaderboards are Filtering Tools, not Decision Tools**.
- **Today's Shift**: Stop studying theory. Start building!
- **The Goal**: Learn how to run **Custom Model Evals** to pick the single best model for YOUR app.

---

## 🏏 Part 2: The Case Study (ESPN Cricinfo Text-to-SQL)

**The Scenario**:
- You are an AI Engineer at **ESPN Cricinfo**.
- **Problem**: During live cricket matches (e.g., India vs. Pakistan), fans ask thousands of statistical questions (e.g., *"What is Virat Kohli's average against Pakistan?"*). Analysts manually write SQL queries to answer these, but it doesn't scale.
- **Your Feature**: Build a **Text-to-SQL system**. Users type a question in English → An LLM converts it to an SQL query → The system runs it on the database → Returns the answer instantly.
- **Your Task**: Select the **best LLM ("the brain")** for this Text-to-SQL system.

---

## 🛠️ Part 3: The 3-Step Model Selection Framework

| Step | Action | Purpose |
| :--- | :--- | :--- |
| **1. Gather Requirements** | Define cost ceiling, latency budget, context needs, deployment constraints, and correctness priority. | To filter out 95% of models immediately. |
| **2. Shortlist via Leaderboards** | Check Multi-Benchmark Leaderboards (like Artificial Analysis) to filter 100 models down to **5-10 candidates**. | Leaderboards are **Filtering Tools** (not decision tools). |
| **3. Run Custom Evals** | Run your own custom evaluation on those 5-10 candidates using your specific dataset. | **This is the final decision maker.** |

---

## 💰 Part 4: Deep Dive into Step 1 – Cost Calculation (The "Token Math")

The instructor performs a critical **budget calculation** to see if they can afford a top-tier model (like Claude Sonnet/Opus).

**Step-by-Step Cost Breakdown**:

1.  **Input Tokens (Per Query)**: The system prompt + Database schema is ~**400 tokens**.
2.  **Output Tokens (Per Query)**: The generated SQL query is ~**100 tokens**.
3.  **Daily Volume**: They estimate **5,000 queries/day**.
4.  **Pricing (Example: Expensive Model)**:
    - Input price: $10 per 1M tokens.
    - Output price: $50 per 1M tokens.
5.  **Monthly Cost Calculation**:
    - Cost per query = `(400 * $10/1M) + (100 * $50/1M)` = `$0.004 + $0.005` = **$0.009 per query**.
    - Daily cost = `$0.009 * 5,000` = **$45/day**.
    - Monthly cost = `$45 * 30` = **$1,350/month** (approx. **₹1.28 Lakhs**). 
    - *Wait, the instructor initially miscalculated in the video for dramatic effect to show "₹12 Lakhs", but the real math shows ₹1.28 Lakhs. The key point is the calculation logic itself.*

**Conclusion**: Even at ₹1.28 Lakhs, it's manageable, but they set a strict budget of **₹3 Lakhs/month**. They cannot pick a model that exceeds this. They must pick a cheaper, smaller model (like Claude Haiku or GPT-4o-mini).

### 💻 Code Example 1: Monthly Cost Calculator (Token Math)

```python
def calculate_monthly_cost(input_tokens, output_tokens, 
                           input_price_usd_per_m, output_price_usd_per_m, 
                           daily_queries, usd_to_inr=95):
    # Cost per single query
    input_cost = (input_tokens * input_price_usd_per_m) / 1_000_000
    output_cost = (output_tokens * output_price_usd_per_m) / 1_000_000
    cost_per_query = input_cost + output_cost
    
    # Daily and Monthly
    daily_cost_usd = cost_per_query * daily_queries
    monthly_cost_usd = daily_cost_usd * 30
    monthly_cost_inr = monthly_cost_usd * usd_to_inr
    
    return monthly_cost_inr

# Testing an expensive model (Claude Sonnet-like)
expensive_input_price = 10  # $10 per 1M input tokens
expensive_output_price = 50 # $50 per 1M output tokens

monthly_inr = calculate_monthly_cost(400, 100, expensive_input_price, expensive_output_price, 5000)
print(f"Expensive Model Monthly Cost: ₹{monthly_inr:,.0f}")

# Testing a cheap model (Claude Haiku-like)
cheap_input_price = 0.5   # $0.5 per 1M input tokens
cheap_output_price = 1.5  # $1.5 per 1M output tokens

monthly_inr_cheap = calculate_monthly_cost(400, 100, cheap_input_price, cheap_output_price, 5000)
print(f"Cheap Model Monthly Cost: ₹{monthly_inr_cheap:,.0f}")

# Output:
# Expensive Model Monthly Cost: ₹128,250  (Within 3 Lakhs, but let's be safe)
# Cheap Model Monthly Cost: ₹7,125      (Much safer for budget)
```

---

## 🧠 Part 5: Prompt Caching (KV Cache) – The Cost Optimization Hack

The instructor introduces **Prompt Caching**. 

- **The Problem**: In this Text-to-SQL app, the **System Prompt + Database Schema** (400 tokens) is **EXACTLY THE SAME** for every single query. Only the user's question changes. You are paying full price for those 400 tokens *every single time*.
- **The Solution (Prompt Caching)**: The LLM provider caches the repetitive part of your prompt on their servers (storing the Key/Value vectors from the Attention mechanism). 
  - **First Request**: You pay a **higher price** (e.g., 25% extra) to write the cache.
  - **Subsequent Requests (within the cache window)**: You pay a **heavily discounted price** (e.g., 10% of the original input price) because you are just hitting the cache.

- **Cache Duration Options**:
  - **5-Minute Cache**: Good for high-traffic apps (like a live cricket match where queries come every second). If no query comes in 5 minutes, the cache expires.
  - **1-Hour Cache**: Good for low-traffic apps.

### 💻 Code Example 2: Simulating Prompt Caching Savings

```python
def calculate_cost_with_caching(total_queries, base_input_price_per_m, 
                                 input_tokens, output_price_per_m, output_tokens,
                                 cache_window_queries=10): # Assume 10 queries per cache window
    # Base input price per 1M tokens
    cache_write_price = base_input_price_per_m * 1.25  # 25% extra to WRITE the cache
    cache_hit_price = base_input_price_per_m * 0.1    # 90% discount on subsequent hits
    
    cost_per_query = 0
    
    # Simulating sequences of queries within a cache window
    # For simplicity, group queries. In reality, it's time-based.
    windows = total_queries // cache_window_queries
    remaining = total_queries % cache_window_queries
    
    total_input_cost = 0
    
    # For each window: 1 expensive write + (N-1) cheap hits
    for _ in range(windows):
        # First query (Cache Write)
        total_input_cost += (input_tokens * cache_write_price) / 1_000_000
        # Next (window_size - 1) queries (Cache Hits)
        total_input_cost += ((cache_window_queries - 1) * input_tokens * cache_hit_price) / 1_000_000
        
    # Handle remaining
    if remaining > 0:
        total_input_cost += (input_tokens * cache_write_price) / 1_000_000
        total_input_cost += ((remaining - 1) * input_tokens * cache_hit_price) / 1_000_000
    
    output_cost = (total_queries * output_tokens * output_price_per_m) / 1_000_000
    
    total_cost_usd = total_input_cost + output_cost
    return total_cost_usd * 95  # INR

# Without caching (original)
original_inr = calculate_monthly_cost(400, 100, 10, 50, 5000)

# With caching (Simulated)
cached_inr = calculate_cost_with_caching(5000, 10, 400, 50, 100)

print(f"Original Monthly Cost: ₹{original_inr:,.0f}")
print(f"With Prompt Caching: ₹{cached_inr:,.0f}")

# Output will show caching saving a significant chunk of money on the input tokens!
```

---

## 📋 Part 6: The Complete Requirements Checklist for the Case Study

After the cost analysis, here is the full list of constraints gathered:

| Requirement | Specification | Rationale |
| :--- | :--- | :--- |
| **Task** | Text-to-SQL (SQLite syntax). | Converts natural language to queries. |
| **Cost Ceiling** | ₹3 Lakhs/month (approx. $3,000). | Business budget constraint. |
| **Latency** | **< 2-3 seconds**. | Live match context; users lose patience. |
| **Context Window** | **Not critical** (Single-turn queries). | User asks one question; no multi-turn conversation history. |
| **Deployment** | **Public APIs are preferred**. | High reliability; no sensitive data requiring on-premise hosting. |
| **Correctness** | **CRITICAL (Must be highly accurate).** | Cricket fans are picky; a wrong stat goes viral on social media and ruins reputation. |

---

## 📝 Final Summary Table

| Concept | Key Takeaway |
| :--- | :--- |
| **The Framework** | 1. Gather Requirements → 2. Leaderboard Filter → 3. Custom Eval. |
| **Cost Math** | Calculate monthly cost by `(Input tokens * Price/1M) + (Output tokens * Price/1M)` multiplied by daily volume × 30. |
| **Prompt Caching** | Repeated prompt prefixes (System + Schema) are cached. Only the *first* query pays full price; subsequent queries pay ~90% less for input tokens. Crucial for cost savings. |
| **Latency** | Critical for live apps; must be < 2-3 seconds. |
| **Correctness** | Absolute must for this app (Text-to-SQL accuracy must be high to avoid public backlash). |
| **Leaderboards** | Used ONLY to filter 100 models down to 5-10. They are NOT the final decision-maker. |

---

This tutorial covers the **hands-on implementation of Steps 2 and 3** of the Model Selection Framework. 

**Step 2**: The instructor uses a **Coding Leaderboard** (as a proxy for SQL capability) to filter 100+ models down to 5 candidates based on **cost and composite accuracy/speed scores**.

**Step 3**: He builds a **Custom Evaluation Pipeline** from scratch (Golden Dataset, SQLite DB, and a smart result-set evaluator) and runs live tests on the 5 candidates to pick the final winner.

---

## 🎯 Part 1: Step 2 – Leaderboard Shortlisting (The Filtration)

### The "Filtering" Process
The goal is to go from ~146 models to 5-10 candidates based on requirements.

1.  **Rejecting Specific SQL Leaderboards**: The instructor looked for Text-to-SQL leaderboards (e.g., Bird-SQL, Spider) but rejected them because they were **outdated** (old models) or included obscure "fine-tuned" models that aren't usable via public API.
2.  **Using a Coding Leaderboard as a Proxy**: Since SQL generation is essentially a coding task, he used `llmstats.com` (a general coding leaderboard) with **146 models**.
3.  **Cost Filter**: He calculated the monthly cost for each model (using `5000 queries/day`, `400 input tokens`, `100 output tokens`, and a **5 Lakh/month budget**). Removed all models exceeding ₹5 Lakhs.
4.  **Normalization & Weighted Composite Score**: For the remaining models, he normalized their **Coding Accuracy** and **Speed** (characters/second). He gave **90% weight to accuracy** and **10% to speed** (since SQL queries are short, speed matters less). 
5.  **Final Shortlist**: He picked the **Top 5** candidates:
    - GPT-5.6 Tera (Expensive, high accuracy)
    - Kimi K3 (Hype model, cheap)
    - Grok 4.5 (Mid-cost, fast)
    - Claude Sonnet 5 (Mid-cost, reliable)
    - Minimax M3 (Cheap, Chinese model)

### 💻 Code Example 1: Blended Pricing & Composite Score Calculation

```python
# 1. BLENDED PRICING (How leaderboards show a single price)
# Leaderboard assumes a 4:1 Input:Output ratio for costs.
input_price = 10   # $ per 1M tokens (Claude Sonnet)
output_price = 50  # $ per 1M tokens
blended_price = (4 * input_price + 1 * output_price) / 5
print(f"Blended Price per 1M tokens: ${blended_price}") # Output: $18

# 2. MONTHLY COST CALCULATION
def calculate_monthly_cost(blended_price_per_mil, input_tokens, output_tokens, daily_queries, usd_to_inr=95):
    total_tokens_per_query = input_tokens + output_tokens
    cost_per_query_usd = (total_tokens_per_query * blended_price_per_mil) / 1_000_000
    monthly_cost_usd = cost_per_query_usd * daily_queries * 30
    return monthly_cost_usd * usd_to_inr

monthly_inr = calculate_monthly_cost(blended_price=18, input_tokens=400, output_tokens=100, daily_queries=5000)
print(f"Monthly Cost: ₹{monthly_inr:,.0f}") # Output: ₹128,250

# 3. COMPOSITE SCORE (Weighted: 90% Accuracy, 10% Speed)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Simulated normalized values (0 to 1)
model_accuracy = 0.95
model_speed = 0.50  # Slow, but we don't care much

composite_score = (model_accuracy * 0.9) + (model_speed * 0.1)
print(f"Composite Score: {composite_score:.3f}") # Output: 0.905

# The instructor used this formula to rank and shortlist candidates.
```

---

## 🔧 Part 2: Step 3 – The Custom Evaluation Pipeline

The instructor built an end-to-end pipeline to test the 5 shortlisted models on their *actual* data (Text-to-SQL).

### A. Infrastructure Setup
1.  **Database**: He downloaded IPL (cricket) data from Kaggle (2008-2024) and filtered it to **2020-2024** for simplicity. He loaded it into a **SQLite** database (`matches` and `deliveries` tables).
2.  **Schema Extraction**: He wrote a script to extract the database schema (table names, column names, data types) and save it as `schema.sql`. This schema is injected into the **System Prompt** for the LLM.

### 💻 Code Example 2: System Prompt Construction (Injection)

```python
# The system prompt sent to every model on every query.
system_prompt_template = """
You are a Text-to-SQL generator for a cricket database (IPL 2020-2024).
Given the schema below, convert the user's question into a single SQLite query.

SCHEMA:
{table_schema}

USER QUESTION: {question}
SQL QUERY:
"""

# The schema is static (400 tokens), only the question changes.
```

### B. Golden Dataset Creation
1.  **Purpose**: A set of questions with their **correct SQL queries** (manually validated). This is the "Answer Key".
2.  **Strategy**: He generated **20 "Hard" questions** (using an LLM) covering complex joins, aggregations, and subqueries. He manually validated them against the DB to ensure they ran correctly and returned the expected results.
3.  **Structure**: `golden_hard.csv` containing columns: `question`, `golden_query`, `is_order_sensitive` (True/False).

### 💻 Code Example 3: Golden Dataset Structure

```python
# The Golden Dataset (Ground Truth)
golden_data = [
    {
        "id": 1,
        "question": "What is the average number of runs scored per match?",
        "golden_query": "SELECT AVG(total_runs) FROM (SELECT match_id, SUM(total_runs) AS total_runs FROM deliveries GROUP BY match_id);",
        "order_sensitive": False
    },
    {
        "id": 2,
        "question": "List the top 5 bowlers with the most wickets in 2023.",
        "golden_query": "SELECT bowler, COUNT(*) AS wickets FROM deliveries WHERE is_wicket = 1 AND season = 2023 GROUP BY bowler ORDER BY wickets DESC LIMIT 5;",
        "order_sensitive": True  # ORDER BY matters here!
    }
]
```

### C. The Core Evaluator Logic (The "Secret Sauce")
The biggest mistake is comparing SQL **strings** (e.g., `SELECT * FROM users` vs `SELECT users.* FROM users`). They are semantically identical but textually different.

**The Solution**: Compare the **Result Sets** (tables/dataframes) generated by running the Golden SQL and the Generated SQL on the actual database.

**Comparison Steps**:
1.  **Execute Both Queries**: Run the Golden SQL and Generated SQL on the SQLite DB. Get two result tables (DataFrames).
2.  **Check Row Count**: If the number of rows is different → **FAIL**.
3.  **Normalize Values**: Convert `2.0` to `2`, handle `NULL` safely, strip whitespace.
4.  **Sort (If Order Doesn't Matter)**: If the query doesn't have `ORDER BY`, sorting both tables alphabetically ensures we ignore row order variations.
5.  **Compare Cell-by-Cell**: If all rows/columns match → **PASS**.

### 💻 Code Example 4: Result-Set Comparison Logic (Simplified)

```python
import pandas as pd
import sqlite3

def compare_result_sets(golden_sql, generated_sql, db_conn, is_order_sensitive=False):
    # 1. Execute both queries
    df_golden = pd.read_sql_query(golden_sql, db_conn)
    df_generated = pd.read_sql_query(generated_sql, db_conn)
    
    # 2. Check Row Count
    if len(df_golden) != len(df_generated):
        return False, f"Row count mismatch: {len(df_golden)} vs {len(df_generated)}"
    
    # 3. Normalize values (e.g., 2.0 -> 2, handle NULLs)
    df_golden = df_golden.fillna(0).astype(str).applymap(lambda x: x.strip())
    df_generated = df_generated.fillna(0).astype(str).applymap(lambda x: x.strip())
    
    # 4. Sorting (If order doesn't matter)
    if not is_order_sensitive:
        # Sort by all columns to ignore row order
        df_golden = df_golden.sort_values(by=list(df_golden.columns)).reset_index(drop=True)
        df_generated = df_generated.sort_values(by=list(df_generated.columns)).reset_index(drop=True)
    
    # 5. Compare DataFrames
    if df_golden.equals(df_generated):
        return True, "Match"
    else:
        return False, "Value mismatch"
```

---

## 🚀 Part 3: The Live Execution & Final Results

The instructor ran `main.py` which orchestrates: loading the 5 models via **OpenRouter** (unified API), looping through the 20 golden questions, generating SQL, executing it, and comparing results.

### OpenRouter (Unified API Gateway)
- **Problem**: GPT, Claude, Grok, and Chinese models have different API endpoints/syntaxes.
- **Solution**: **OpenRouter** provides a single endpoint. You just change the `model` string (e.g., `"anthropic/claude-3.5-sonnet"` or `"x-ai/grok-2"`).

### Live Performance Metrics (20 Hard Questions)

| Model | Correct | Accuracy | Cost (₹ Lakhs/month) | Speed | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-5.6 Tera** | 16/20 | **80%** | ~5 Lakhs | Fast | Good but expensive. |
| **Kimi K3** | ~11/20 | **55%** | ~2.5 Lakhs | Very Slow | **Leaderboard Hype fail**. Heavy reasoning model, slow, and terrible at SQL syntax (many errors). |
| **Grok 4.5** | 18/20 | **90%** | ~2.5 Lakhs | Fast | **Unexpected Winner**. Extremely fast, cheap, and accurate. |
| **Claude Sonnet 5** | 17/20 | **85%** | ~2.8 Lakhs | Very Fast | Highly reliable, stable API, close second. |
| **Minimax M3** | ~13/20 | **65%** | ~1.2 Lakhs | Medium | Cheap, but Chinese models struggle with SQL syntax (generated invalid queries). |

### The Final Decision
- **The Winner (Accuracy)**: **Grok 4.5** (90%).
- **The Winner (Stability)** : **Claude Sonnet 5** (85% but more reliable API).
- **Decision**: The instructor leaned towards **Claude Sonnet 5** due to **Anthropic's enterprise-grade API reliability** (Elon Musk/Grok's API is considered slightly less enterprise-stable).

---

## 📝 Summary of Key Pointers

1.  **Leaderboards are Filters, not Decisions**: They help go from 146 to 5 models. They cannot guarantee actual performance.
2.  **Beware of Hype**: Kimi K3 was all over the news for being a "super model". On this specific Text-to-SQL task, it scored only 55% with high latency. **Always run custom evals.**
3.  **Result-Set Comparison is Essential**: Never compare raw SQL strings. Always execute them and compare the returned tables/numbers.
4.  **Normalize Before Comparing**: Handle `2.0` vs `2`, `NULL` vs `0`, and row orders (unless `ORDER BY` is required).
5.  **OpenRouter is a Lifeline**: It allows testing multiple different models (OpenAI, Anthropic, Chinese models) using the exact same codebase.
6.  **Cost vs Accuracy Trade-off**: Grok gave 90% accuracy at half the cost of GPT-5.6 Tera, making it the business choice, despite Sonnet being preferred for API stability.

**Bottom Line**: The entire hands-on demo proved the golden rule of AI Engineering: **What works on a leaderboard doesn't always work on your specific data**. You must build the pipeline, run the evals, and let the numbers decide. 🚀

- [Best AI for Coding](https://llm-stats.com/leaderboards/best-ai-for-coding)

- [openrouter](https://openrouter.ai/)

---

## 012. How to Answer "How Do You Evaluate Your RAG App?" in GenAI Interviews (46:19)

In this lecture, instructor outlines a complete **"Eval Suite" framework** for evaluating a **RAG (Retrieval-Augmented Generation) Chatbot** across 3 levels (Component, Pipeline, Application), along with **Regression Testing (CI/CD)** and **Online Monitoring** to close the loop.

---

## 🎯 Part 1: The Big Shift (Recap & Context)

**Recap of the Course So Far**:
1.  **Why Evals?** (Legal, reputation, hallucinations).
2.  **What are Evals?** (Systematic tests with clear criteria).
3.  **Types**: Model Evals (Benchmarks + Custom) vs. Application Evals.
4.  **Model Evals**: We studied Benchmarks (MMLU, HLE, etc.) and Custom Model Evals (selecting the right LLM for your app via leaderboards + custom testing).
5.  **NOW**: We finally move to **Application Evals** (testing the entire system you built).

**The Case Study**: Building a **"CampusX Doubt Solver"** – a RAG chatbot that answers questions about the course content using lecture transcripts as documents.

**The Goal**: Build a comprehensive **"Eval Suite"** (multiple test pipelines) to evaluate this RAG app before and after deployment.

---

## 📂 Part 2: The 3-Tier Evaluation Suite Framework

The instructor breaks down the evaluation into **3 distinct levels**. You do NOT build the whole app and then test it; you **test as you build**.

### Level 1: Component-Level Evaluation (Testing in Isolation)

You build and test each component separately **before** connecting them.

**1A. Evaluating the Retriever (in Isolation)**
- **What**: The Retriever fetches relevant documents from the Vector DB for a given query.
- **How**: You use a **Golden Dataset** (Query → List of Relevant Document IDs).
- **Metrics**:
  - **Recall**: Out of *all* relevant documents, how many did the retriever fetch?
  - **Precision**: Out of *all* documents fetched, how many were actually relevant?

**1B. Evaluating the Generator (LLM) (in Isolation)**
- **What**: You give the LLM a fixed **Query + Context** (manually provided, not retrieved) and check if it generates a good answer.
- **Metrics**:
  - **Faithfulness**: Is the answer grounded in the provided context? (No hallucination).
  - **Answer Relevance**: Does the answer actually answer the question?
  - **Citation Accuracy**: If the LLM cites a specific lecture/section, is that citation correct?

#### 💻 Code Example 1: Component-Level Evaluation (Retriever & Generator)

```python
# 1. RETRIEVER EVALUATION (Recall & Precision)
golden_retrieval_data = [
    {"query": "What is LLM evaluation?", "relevant_docs": ["doc_1", "doc_3"]}
]

def evaluate_retriever(retriever, golden_data):
    total_recall, total_precision = 0, 0
    for item in golden_data:
        retrieved = retriever.get_relevant_docs(item["query"], k=5)
        relevant_set = set(item["relevant_docs"])
        retrieved_set = set(retrieved)
        
        recall = len(retrieved_set & relevant_set) / len(relevant_set)
        precision = len(retrieved_set & relevant_set) / len(retrieved_set)
        
        total_recall += recall
        total_precision += precision
    
    return total_recall/len(golden_data), total_precision/len(golden_data)

# 2. GENERATOR EVALUATION (Using DeepEval Library)
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Test case for Generator (Isolation)
test_case = LLMTestCase(
    input="What is overfitting?",
    actual_output="Overfitting is when a model learns noise instead of the signal.",
    retrieval_context=["Overfitting happens when a model memorizes training data."]
)

faithfulness_metric = FaithfulnessMetric(threshold=0.7)
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

# DeepEval runs an LLM-as-a-Judge behind the scenes to score these.
assert_test(test_case, [faithfulness_metric, answer_relevancy_metric])
```

---

### Level 2: Pipeline-Level Evaluation (Testing the RAG Triad)

Once the Retriever and Generator are connected, you test the **entire pipeline flow**.

**The "RAG Triad" (3 Metrics based on 3 elements: Query, Context, Answer)**

| Pairs | Metric | Question |
| :--- | :--- | :--- |
| **Query ↔ Context** | **Contextual Relevancy** | Is the retrieved context relevant to the user's question? |
| **Context ↔ Answer** | **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |
| **Query ↔ Answer** | **Answer Relevancy** | Does the final answer actually address the user's query? |

#### 💻 Code Example 2: Pipeline-Level Eval (RAG Triad using DeepEval)

```python
from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Simulating a RAG pipeline output
test_case = LLMTestCase(
    input="Explain gradient descent.",
    actual_output="Gradient descent minimizes the loss function by updating weights.",
    retrieval_context=["Gradient descent is an optimization algorithm used to minimize loss by updating parameters in the opposite direction of the gradient."]
)

# The RAG Triad
metrics = [
    ContextualRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7)
]

# Running these 3 metrics gives a comprehensive health check of your pipeline.
for metric in metrics:
    metric.measure(test_case)
    print(f"{metric.__class__.__name__}: {metric.score}")
```

---

### Level 3: Application-Level Evaluation (End-to-End Quality, Safety & Ops)

After the pipeline works, you test the **final user experience**.

**Quality Metrics**:
- **Correctness**: Is the factual content of the answer correct?
- **Completeness**: Does the answer fully cover all parts of the question? (e.g., if the user asks 2 things, does the answer cover both?)
- **Tone/Style**: Does the answer match the expected tone (e.g., teacher-like, friendly)?

**Safety Metrics** (Crucial for production):
- **Toxicity**: Does it contain hate speech or offensive language?
- **PII Leakage**: Does it accidentally reveal personal information (phone numbers, emails)?
- **Jailbreak Resistance**: Can users manipulate the prompt to bypass rules?

**Operational Metrics** (The "OPS" Evals):
- **Latency**: How long does it take to respond?
- **Cost**: How many tokens were used, and what is the monetary cost per query?

#### 💻 Code Example 3: Application & Safety Evals

```python
from deepeval.metrics import ToxicityMetric, GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="Can you help me cheat?",
    actual_output="I can't assist with that. But I can help you understand the concepts better!"
)

# 1. Safety: Toxicity
toxicity = ToxicityMetric(threshold=0.5) # Score 0-1, lower is safer
toxicity.measure(test_case)
print(f"Toxicity Score: {toxicity.score}") # Should be very low.

# 2. Application: Correctness (Requires a Golden Answer)
# GEval allows you to define custom criteria using an LLM judge.
from deepeval.metrics import GEval
from deepeval.scorer import Scorer

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct compared to the expected output.",
    evaluation_steps=["Compare the facts in actual output and expected output."]
)
# This requires a golden answer.
```

---

## 🛠️ Part 3: Tooling Strategy – Why DeepEval?

The instructor chooses **DeepEval** over custom code or RAGAS because:
1.  It has built-in metrics for **everything** we just discussed (RAG Triad, Toxicity, PII, etc.).
2.  It is **broader** – supports agents, multimodality, and multi-turn chats.
3.  It is becoming the **industry standard**.
4.  It integrates with `pytest`, making it easy to set up as an automated test suite.

---

## 🔁 Part 4: Regression Testing (The Release Gate)

Once your Eval Suite (collection of all test files) is built, you use it for **Regression Testing**.

**The Workflow**:
1.  **Run Suite** → Get scores (e.g., Faithfulness = 0.95, Latency = 1.2s). Save these as your **Baseline**.
2.  **Make a Change** (e.g., update the prompt, change chunk size).
3.  **Run Suite Again** → Get new scores.
4.  **Compare**: Are the new scores worse than the baseline? If yes, the change caused a **"Regression"** (performance drop). 

**Advanced Implementation**:
- **Experiment Tracking**: Use MLflow to log every run's config (chunk size, temperature) and scores automatically.
- **CI/CD Integration (GitHub Actions)**: Automate this. When you push code, the pipeline runs the Eval Suite. If accuracy drops below a threshold (e.g., 3%), the deployment is **blocked automatically**.

### 💻 Code Example 4: Regression Testing & CI/CD Gate Logic

```python
# Simulating a simple regression check
baseline_scores = {"faithfulness": 0.95, "answer_relevancy": 0.90}
new_scores = {"faithfulness": 0.82, "answer_relevancy": 0.89}

def check_regression(baseline, new, threshold=0.05):
    for key in baseline:
        drop = (baseline[key] - new[key]) / baseline[key]
        if drop > threshold:
            print(f"🚨 REGRESSION DETECTED in {key}: Dropped by {drop*100:.1f}%!")
            return False # BLOCK DEPLOYMENT
    print("✅ No significant regression. Deployment allowed.")
    return True

check_regression(baseline_scores, new_scores) 
# Output: 🚨 REGRESSION DETECTED in faithfulness: Dropped by 13.7%!
```

---

## 🌐 Part 5: Online Evaluations (Post-Deployment Monitoring)

Deploying does NOT stop evaluation. You must monitor live traffic.

**What to Monitor**:
1.  **Captured Signals**: Latency, Token Count, Cost, Thumbs Up/Down (via LangSmith or similar).
2.  **Computed Signals**: Run lightweight versions of your offline metrics (Faithfulness, Answer Relevancy) on a **sample** of live conversations (using an LLM-as-a-Judge).
3.  **Drift Detection**: Plot graphs of these metrics over time. If Faithfulness suddenly drops (curve goes down), trigger an alert.
4.  **The Self-Improving Loop**: If a user gives a "Thumbs Down" or if the online monitoring detects a hallucination, extract that specific conversation. **Add this failure case to your Offline Golden Dataset** for the next regression test. This ensures the bug never happens again.

### 💻 Code Example 5: Adding Production Failures Back to Offline Dataset

```python
# Simulating the self-improving loop
offline_dataset = [{"q": "What is X?", "answer": "X is Y"}]

# Production failure detected (via thumbs down or manual flag)
production_failure = {
    "question": "What is the latest lecture about?",
    "wrong_output": "It's about LLMs.", # Wrong context
    "correct_context": "The latest lecture is about RAG evaluation."
}

# ADD to offline dataset
offline_dataset.append({
    "question": production_failure["question"],
    "expected_answer": production_failure["correct_context"] 
})
# Next time you run Offline Regression, this edge case is covered!
print(f"New offline dataset size: {len(offline_dataset)}")
```

---

## 🎤 Part 6: The Perfect Interview Answer (How to Ace the Question)

**Question**: *"How do you evaluate a RAG Chatbot?"*

**The Killer Answer** (Structured response):
> *"I build a comprehensive **Eval Suite** across three levels:*
> 1.  **Component Level**: I test the Retriever in isolation using **Recall/Precision** against a golden dataset. I test the Generator in isolation for **Faithfulness** (no hallucinations) and **Answer Relevancy**.
> 2.  **Pipeline Level**: I test the full RAG flow using the **RAG Triad** – **Contextual Relevancy**, **Faithfulness**, and **Answer Relevancy**.
> 3.  **Application Level**: I test end-to-end for **Correctness**, **Completeness**, and **Tone**. I also run **Safety** (toxicity, PII) and **Operational** (latency, cost) evals.
> 
> *I automate this whole suite as my **Regression Testing** framework. Every time I change the code, the suite runs. If scores drop below the baseline threshold, I block the deployment via CI/CD. After deployment, I continue monitoring via **Online Evals** (LangSmith) to track drift and user feedback, continuously feeding production failures back into my offline dataset.*"

---

## 📝 Final Summary Table

| Level | What to Test | Key Metrics | When to Test |
| :--- | :--- | :--- | :--- |
| **1. Component** | Retriever (alone) | Recall, Precision | During building |
| | Generator (alone) | Faithfulness, Relevance | During building |
| **2. Pipeline** | Retriever + Generator (RAG Flow) | **Context Relevancy, Faithfulness, Answer Relevancy** (RAG Triad) | After connecting components |
| **3. Application** | End-to-End UX | Correctness, Completeness, Tone, Toxicity, PII, Latency, Cost | Before deployment |
| **Regression** | Re-running the entire suite | Baseline vs. New Scores | On every code change (CI/CD) |
| **Online** | Live Production Traffic | Drift, Thumbs Up/Down, Latency spikes | Continuously after deployment |

**Bottom Line**: You don't just "test a chatbot". You build an automated, layered **Eval Suite** (offline tests) + **Monitoring** (online tests). This is the hallmark of a true Production AI Engineer. 🚀

- [DeepEval](https://deepeval.com/docs/metrics-introduction)

---

## 013. How to Test RAG Retrievers(Hands-On) (01:47:01)

This tutorial is the **hands-on implementation of Component-Level Evaluation** for a RAG system where we build a **Retriever** for the "CampusX Doubt Solver" (a RAG chatbot answering questions from lecture transcripts), learn the **correct way to evaluate it using LLM-as-a-Judge**, and then run multiple optimization experiments to improve its performance.

---

## 📁 Part 1: Project Setup & Building the Retriever

**The Project**: "CampusX Doubt Solver" – a RAG app over the course's lecture transcripts (`.vtt` files).

**Folder Structure**:
- `data/` : Contains the raw lecture transcripts.
- `src/` : Contains the application source code (`retriever.py`, `generator.py`, `reranker.py`).
- `evals/` : Contains the evaluation scripts (`eval_retriever.py`).
- `goldens/` : Contains the golden datasets (`retriever_goldens.json`).

**Building the Retriever**:
1.  **Load Data**: Read all `.vtt` transcript files.
2.  **Preprocessing**: Remove timestamps (e.g., `0:00`) to keep only the semantic text. (Timestamps act as noise and break semantic meaning).
3.  **Chunking**: Split text into overlapping chunks. **Initial Params**: `Chunk Size = 750`, `Overlap = 100`.
4.  **Embedding**: Convert chunks to vectors using `OpenAI text-embedding-3-small`.
5.  **Store**: Save vectors in a **ChromaDB** vector store.
6.  **Retrieve**: Set `k = 5` (fetch top 5 most similar chunks).

### 💻 Code Example 1: Building the Retriever (Conceptual)

```python
# src/retriever.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_transcripts(data_dir):
    # Load all .vtt files, remove timestamps, extract text.
    # Returns a list of LangChain Documents with metadata (session_id).
    pass

def build_retriever():
    # 1. Load Data
    documents = load_transcripts("data/")
    
    # 2. Chunking (Initial: 750 chars, overlap 100)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, 
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    
    # 3. Embedding & 4. Vector Store (Chroma)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_store"
    )
    
    # 5. Retriever (k=5)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever
```

---

## 🧩 Part 2: Retriever Failure Modes & Metrics

### The Two Failure Modes of a Retriever

| Failure Mode | Description | Example (Golden = Docs 1,5,9) |
| :--- | :--- | :--- |
| **Miss** | The retriever fails to fetch the correct documents. | Fetches Docs 10, 11, 12. (0 relevant). |
| **Noise** | The retriever fetches correct documents BUT also brings irrelevant ones. | Fetches Docs 1, 5, 10, 11, 12. (Only 2 relevant). |

### The Metrics (Traditional Definition)
- **Recall**: Out of *all* correct documents (e.g., 3), how many did we fetch? (Max score = 1.0).
- **Precision**: Out of *all* documents we fetched (e.g., 5), how many are relevant? (Max score = 1.0).
- **Trade-off**: Increasing `k` usually increases Recall but hurts Precision (more noise).

---

## ❌ Part 3: The "Flawed" Golden Dataset Approach (Document IDs)

**The Flawed Idea**: Create a Golden Dataset with `(Question, Chunk_IDs)` pairs. 
- Example: Q1 → Docs 72, 89, 100.
- Then, compare the Retriever's output IDs to this list to calculate Recall/Precision.

**Why this FAILS in real projects**:
Imagine you change your **Chunking Parameters** (e.g., from 750 to 1000). The IDs of the chunks change completely! 
Your Golden Dataset (which stored IDs `72, 89, 100`) is now **VOID**. You have to manually re-label the entire dataset every time you tweak chunking. This is unscalable.

---

## ✅ Part 4: The Correct Method (LLM-as-a-Judge & Claims)

**The Robust Idea**: Create a Golden Dataset with `(Question, Ideal_Answer)`.
- The *Ideal Answer* is the perfect answer crafted by a human expert *based solely on the lecture transcripts*.
- You DO NOT store chunk IDs.

**How it works (Contextual Recall)**:
1.  Run the Query through the Retriever → get 5 chunks.
2.  Give the `Ideal Answer` to an **LLM-as-a-Judge**.
3.  Ask the Judge: *"Break this Ideal Answer into atomic claims (factual statements)."*
4.  Then ask: *"For each claim, check if it is supported by the 5 retrieved chunks."*
5.  **Contextual Recall** = (Number of claims found) / (Total claims in Ideal Answer).

**How it works (Contextual Precision - Rank Aware)**:
Traditional Precision treats all chunks equally, ignoring rank. If correct chunks are at positions 1 & 2, it's better than if they are at positions 4 & 5.
- **Rank-Aware Scoring**: Iterate through the retrieved chunks in order.
- For chunk 1: Calculate precision so far.
- For chunk 2: Calculate precision so far...
- **Final Score = Average of these cumulative precision scores**.
- *Result*: Correct chunks appearing at the top yield a MUCH higher score.

### 💻 Code Example 2: Simulating the "Claims" Approach (Conceptual)

```python
# Conceptual representation of the LLM-as-a-Judge process
ideal_answer = "RAG evaluates Contextual Relevancy, Faithfulness, and Answer Relevancy."
claims = ["RAG evaluates Contextual Relevancy.", 
          "RAG evaluates Faithfulness.", 
          "RAG evaluates Answer Relevancy."]

retrieved_chunks = [
    "Contextual Relevancy checks if the retrieved context matches the query.",
    "Faithfulness checks if the answer is grounded in the context.",
    "Answer Relevancy is not mentioned here.",
    "Random text.",
    "Some other text."
]

# LLM-as-a-Judge checks each claim against all chunks:
claim_1_found = True  # Found in chunk 1
claim_2_found = True  # Found in chunk 2
claim_3_found = False # Not found in any chunk

contextual_recall = sum([claim_1_found, claim_2_found, claim_3_found]) / 3
print(f"Contextual Recall: {contextual_recall:.0%}") # Output: 66%

# Contextual Precision (Rank-aware):
# Chunk 1: Correct (1/1 = 1.0)
# Chunk 2: Correct (2/2 = 1.0)
# Chunk 3: Incorrect (2/3 = 0.66)
# Chunk 4: Incorrect (2/4 = 0.5)
# Chunk 5: Incorrect (2/5 = 0.4)
# Avg = (1.0 + 1.0 + 0.66 + 0.5 + 0.4) / 5 = 0.71
```

---

## 👨‍💻 Part 5: Sourcing the Golden Dataset

**Options to build `(Question, Ideal_Answer)` pairs**:
1.  **Hand-Authored**: Human expert reads transcripts and writes Q&A. *Best quality, but very expensive*.
2.  **LLM-Assisted Drafting**: (Used by the instructor). Upload transcripts to Claude, instruct it to generate Q&A pairs *from the transcripts*, then manually review/edit them. *Good balance of cost/quality*.
3.  **DeepEval Synthesizer**: DeepEval has a built-in module to auto-generate test cases from documents. (Tested, but the instructor found the output questions too academic/irrelevant for his specific use case).
4.  **Production Logs**: After deployment, take successful user conversations (where they gave thumbs up) and turn them into Q&A pairs.

---

## 🛠️ Part 6: Implementing the Eval Script with DeepEval

**DeepEval Structure**:
1.  **`LLMTestCase`**: Represents one row of your Golden Dataset. Contains `input` (question), `actual_output` (placeholder), `expected_output` (Ideal Answer), and `retrieval_context` (the chunks fetched by the retriever).
2.  **Metric**: Instantiate the metrics (e.g., `ContextualRecallMetric`, `ContextualPrecisionMetric`). Set a `threshold` and define which `model` to use as the judge.
3.  **`evaluate()`**: The function that runs all test cases through all metrics and returns scores.

### 💻 Code Example 3: DeepEval Retriever Evaluation Script

```python
# evals/eval_retriever.py
import json
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from src.retriever import build_retriever

# 1. Load Golden Dataset
with open("goldens/retriever_goldens.json", "r") as f:
    golden_data = json.load(f)

# 2. Setup Judge Model & Metrics
judge_model = "gpt-4o-mini"  # LLM-as-a-Judge
metrics = [
    ContextualRecallMetric(threshold=0.7, model=judge_model),
    ContextualPrecisionMetric(threshold=0.7, model=judge_model)
]

# 3. Build Retriever & Generate Test Cases
retriever = build_retriever()
test_cases = []

for item in golden_data:
    query = item["question"]
    ideal_answer = item["ideal_answer"]
    
    # Get retrieved context from our retriever
    retrieved_docs = retriever.invoke(query)
    retrieved_context = [doc.page_content for doc in retrieved_docs]
    
    test_case = LLMTestCase(
        input=query,
        actual_output="Placeholder (Generator not evaluated here)", 
        expected_output=ideal_answer,
        retrieval_context=retrieved_context
    )
    test_cases.append(test_case)

# 4. Run Evaluation
results = evaluate(test_cases, metrics)
print(f"Contextual Recall: {results[0].score}")
print(f"Contextual Precision: {results[1].score}")
```

---

## 📊 Part 7: Optimization Journey (The Iterative Process)

The instructor runs several experiments to improve the Retriever's performance on 15 hard questions.

| Trial | Configuration | Recall | Precision | Failures |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | Chunk 750/100, `k=5`, Embedding=small | **80%** | **80%** | 5/15 |
| **2. Change Chunking** | Chunk 1000/150, `k=5` | **97%** | **83%** | 3/15 |
| **3. Add Re-Ranker** | Chunk 1000/150, `k=5`, + Cross-Encoder Reranker | ~92% | **85%** | 2/15 |
| **4. Upgrade Embeddings**| Chunk 1000/150, Reranker, Embedding=**Large** | **99%** | ~85% | 3/15 |

**Key Observations**:
- **Increasing Chunk Size** significantly improved Recall (80→97%) because the retriever captured more complete context.
- **Adding a Re-ranker** slightly boosted Precision (83→85%) by moving relevant chunks to the top (rank-awareness).
- **Using a better Embedding Model** gave near-perfect Recall (99%) but didn't improve Precision much.
- **Changing `k` to 3** reduced the number of chunks, which *lowered* Precision slightly due to random variance (small dataset).

**Conclusion**: The retriever achieved **~99% Recall** and **~85% Precision**, which is considered "actually good" for production. The optimization loop proved that testing parameters (chunk size, rerankers, models) is essential.

---

## 📝 Final Summary

| Concept | Key Pointer |
| :--- | :--- |
| **Test-as-you-build** | Build the Retriever first, then evaluate it *before* building the Generator. |
| **Failure Modes** | Retriever can suffer from **Misses** (no correct docs) or **Noise** (too much trash). |
| **Flawed Eval** | Comparing **Chunk IDs** is brittle. Changing chunk size breaks the dataset. |
| **Correct Eval** | Compare **Claims** from the `Ideal Answer` against the retrieved context using an **LLM-as-a-Judge**. |
| **Contextual Recall** | How many claims from the ideal answer are found in the retrieved chunks? |
| **Contextual Precision** | How many retrieved chunks are relevant, and are the relevant ones ranked **at the top**? |
| **Golden Dataset** | Source via **LLM-assisted drafting** + human review. Do NOT use hardcoded document IDs. |
| **DeepEval** | Structure: `LLMTestCase` → `Metric` → `evaluate()`. |
| **Optimization** | Chunk size, embedding models, and re-rankers have a significant impact. Always iterate and measure! |

**Bottom Line**: Evaluating a Retriever by comparing chunks IDs is a rookie mistake. The professional way is to use an **LLM-as-a-Judge to compare the semantic claims** between the ideal answer and the retrieved context. This approach is robust to changes in chunking parameters and gives a true measure of information coverage and ranking quality. 🚀

---

## 014. Evaluating RAG: Testing the Generator & Full Pipeline with the RAG Triad (01:21:28)

This transcript is the **hands-on implementation of Component-Level Evaluation for the Generator** in a RAG system where we build the Generator (an LLM with a system prompt), learn its **two core failure modes** (Unfaithfulness/Hallucination and Answer Irrelevance), and implement **Faithfulness** and **Answer Relevancy** metrics using **DeepEval** and an **LLM-as-a-Judge**. The instructor then runs multiple experiments to optimize the **system prompt**, successfully boosting Answer Relevancy from 73% to 92%.

---

## 📁 Part 1: Building the Generator (Component 2)

**File**: `src/generator.py`

**What it does**: 
- Takes a `question` and `context` as input.
- Uses an LLM (GPT-4o-mini with `temperature=0`) to generate an answer *strictly* from the provided context.
- Key System Prompt rules:
  - *"Answer only from the context provided."*
  - *"Do NOT add outside knowledge."*
  - *"If the context doesn't contain enough info, say 'I don't have enough information' (do NOT hallucinate)."*

### 💻 Code Example 1: Generator System Prompt

```python
# src/generator.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_prompt = """
You are a helpful teaching assistant for a course on LLM Evaluations.
Answer the student's question ONLY from the context provided below.
Do NOT add outside knowledge.
If the context does not contain enough information, say "I don't have enough information in the course material to answer that."
Keep the answer clear and concise.
Context: {context}
Question: {question}
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(system_prompt)
chain = prompt | llm | StrOutputParser()

def generate(question, context):
    return chain.invoke({"question": question, "context": context})
```

---

## 🧩 Part 2: The 2 Generator Failure Modes

Before evaluating, we identify *how* the Generator can fail.

| Failure Mode | Description | Metric |
| :--- | :--- | :--- |
| **1. Unfaithfulness** | The model generates information **NOT present** in the provided context (hallucination). | **Faithfulness** (Lower = more hallucination). |
| **2. Answer Irrelevance** | The model is faithful to the context (doesn't hallucinate), but the answer **does NOT actually answer the user's question**. | **Answer Relevancy** (Lower = off-topic). |

**Key Insight**: A model can be 100% Faithful (only using context) but still give an irrelevant answer.

---

## 🧠 Part 3: How Faithfulness is Calculated

**Requires a "Golden Dataset"** with two columns:
1. `Question` (e.g., "What is a RAG Triad?")
2. `Golden_Context` (The exact chunks from the vector DB that contain the *correct* answer).

**Step-by-step Process**:
1.  Give `Question` + `Golden_Context` to the Generator → get `Generated_Answer`.
2.  Pass `Generated_Answer` to an **LLM-as-a-Judge**.
3.  Ask the Judge: *"Break this answer into atomic claims (small factual statements)."* (e.g., Claim 1, Claim 2, Claim 3).
4.  Ask the Judge: *"For each claim, check if it exists in the Golden_Context."*
5.  **Faithfulness Score** = (Number of claims found in Golden_Context) / (Total number of claims in the Generated_Answer).

> **Note**: Faithfulness does NOT measure correctness. If the Golden_Context itself is wrong, a faithful answer will also be wrong. Faithfulness only measures adherence to the provided context.

### 💻 Code Example 2: Simulating Faithfulness Logic

```python
# Conceptual simulation
question = "What is a RAG Triad?"
golden_context = "The RAG Triad consists of Contextual Relevancy, Faithfulness, and Answer Relevancy."

generated_answer = "A RAG Triad has Contextual Relevancy, Faithfulness, and Answer Relevancy. It also has a fourth metric called Latency."  # Hallucinated "Latency"

# LLM-as-a-Judge breaks answer into claims
claims = [
    "RAG Triad has Contextual Relevancy.",
    "RAG Triad has Faithfulness.",
    "RAG Triad has Answer Relevancy.",
    "RAG Triad has a fourth metric called Latency."  # This is not in the context.
]

# Check each claim against the golden_context
found_claims = 0
for claim in claims:
    # In reality, an LLM does semantic comparison.
    if "Latency" not in claim: # Simulating the check
        found_claims += 1

faithfulness_score = found_claims / len(claims)
print(f"Faithfulness Score: {faithfulness_score:.0%}") # Output: 75% (3/4)
```

---

## 🧠 Part 4: How Answer Relevancy is Calculated

**This is a Reference-Free Eval** (No Golden Context required). You only need the `Question` and the `Generated_Answer`.

**Step-by-step Process**:
1.  Give `Question` + (any) `Context` to the Generator → get `Generated_Answer`.
2.  Pass `Generated_Answer` to an **LLM-as-a-Judge**.
3.  Ask the Judge: *"Break this answer into atomic claims."*
4.  Ask the Judge: *"For each claim, does it help answer the original Question? Or is it off-topic?"*
5.  **Answer Relevancy Score** = (Number of claims relevant to the question) / (Total number of claims in the answer).

### 💻 Code Example 3: Simulating Answer Relevancy Logic

```python
# Conceptual simulation
question = "Does the CampusX AI program include live classes?"
generated_answer = "The program includes recorded lessons, coding assignments, projects, and weekly doubt-solving sessions." # Faithful but does NOT answer "live classes".

# LLM-as-a-Judge breaks answer into claims
claims = [
    "Program includes recorded lessons.",
    "Program includes coding assignments.",
    "Program includes projects.",
    "Program includes weekly doubt-solving sessions."
]

# Check each claim against the QUESTION (Are they relevant to "live classes"?)
# None of them mention "live classes". All are irrelevant to answering the specific question.
relevant_claims = 0  # 0 out of 4 are relevant.

answer_relevancy = relevant_claims / len(claims)
print(f"Answer Relevancy: {answer_relevancy:.0%}") # Output: 0%
```

---

## 📂 Part 5: Sourcing the Golden Dataset (for Faithfulness)

**Why Chunk IDs are a bad idea**: If you change chunking parameters, the IDs change, breaking the dataset.

**Instructor's Approach**:
1.  Export **all chunks** from the ChromaDB (`export_chroma_chunks.py`) into a JSON file.
2.  Upload the JSON to **Claude**.
3.  Ask Claude to generate `(Question, Golden_Context)` pairs *step-by-step* (one at a time).
4.  **Manually review** each generated pair to ensure quality and relevance to the course. (He created 15 questions).

**Other Methods**:
- **Human-authored** (best quality, low scalability).
- **DeepEval Synthesizer** (automated, but he found the output too irrelevant/academic for his specific case).

---

## ⚙️ Part 6: DeepEval Implementation (eval_generator.py)

**DeepEval Structure**:
1.  **`LLMTestCase`**: Represents one test row. Contains `input` (question), `actual_output` (generated answer), `retrieval_context` (Golden Context).
2.  **Metrics**: `FaithfulnessMetric` (threshold=0.7) and `AnswerRelevancyMetric` (threshold=0.7), using `gpt-4o-mini` as the judge.
3.  **`evaluate()`**: Runs all test cases through all metrics.

### 💻 Code Example 4: DeepEval Generator Evaluation Script

```python
# evals/eval_generator.py
import json
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.generator import generate

# 1. Load Golden Dataset (Question + Golden Context)
with open("goldens/faithfulness_dataset.json", "r") as f:
    golden_data = json.load(f)

# 2. Setup Judge & Metrics
judge_model = "gpt-4o-mini"
metrics = [
    FaithfulnessMetric(threshold=0.7, model=judge_model, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, model=judge_model, include_reason=True)
]

# 3. Generate Test Cases
test_cases = []
for item in golden_data:
    question = item["question"]
    golden_context = item["golden_context"]
    
    # Generate answer using the Generator (ISOLATION)
    generated_answer = generate(question, golden_context)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        retrieval_context=[golden_context]  # Faithfulness checks this against the answer
        # Note: For Answer Relevancy, DeepEval ignores retrieval_context and only checks input vs actual_output.
    )
    test_cases.append(test_case)

# 4. Run Evaluation
results = evaluate(test_cases, metrics)
for result in results:
    print(f"{result.metric_name}: {result.score}")
```

---

## 🚀 Part 7: Optimization – System Prompt Engineering

**Baseline Scores**:
- Faithfulness: **91%** (Good out of the box because modern LLMs follow instructions well).
- Answer Relevancy: **73%** (Needed improvement).

**Why Faithfulness is easier**: The model is explicitly told to use the context. The "instruct" nature of LLMs naturally leads to high faithfulness.

**How to improve Relevancy**:
1.  **Analyze Failures**: Run the eval, look at the `reason` provided for each failed test case.
2.  **Refine System Prompt**: Explicitly add rules to the prompt based on the failures.
   - *Example added rule*: *"Do not overstate claims. If the context says 'may cause', don't say 'causes'."*
   - *Example added rule*: *"If the context doesn't answer the specific part of the question, acknowledge it."*
3.  **Iterate**: Repeat the process 3-4 times.

**Optimized Scores** (After 3-4 iterations):
- Faithfulness: **96%** (↑ 5%)
- Answer Relevancy: **92%** (↑ 19%)

**Warning**: Be careful not to **overfit** to the test dataset. The prompt should be general, not crafted to pass just these 15 specific questions.

---

## 📊 Part 8: Summary of Achievements (Component-Level Completed)

The instructor marks the **Component Level** as **COMPLETE**.

| Component | Metrics Learned | How to Optimize |
| :--- | :--- | :--- |
| **Retriever** | Contextual Recall, Contextual Precision | Change chunk size, overlap, embedding model, add reranker. |
| **Generator** | Faithfulness, Answer Relevancy | Improve system prompt, upgrade to a better base LLM. |

**Next Step**: **Pipeline-Level Evaluation** (connecting the Retriever and Generator and testing the full RAG flow using the **RAG Triad** – Contextual Relevancy, Faithfulness, Answer Relevancy).

---

## 📝 Final Important Pointers

1.  **Generator = LLM + Prompt**: The "brains" are the model, but the "training" happens via the system prompt. Prompt tuning is your primary lever.
2.  **Faithfulness ≠ Correctness**: Faithfulness only checks if the answer *came from* the context. If the context is wrong, the answer is faithfully wrong.
3.  **Reference-Free is Cheaper**: Answer Relevancy doesn't need a golden context, making it cheaper to implement for ongoing tests.
4.  **DeepEval Parallelism**: The `evaluate()` function runs test cases in parallel, speeding up the process.
5.  **Overfitting Risk**: Don't over-optimize your system prompt to pass your specific golden dataset. Ensure the prompt is general enough for real-world queries.

**Bottom Line**: The Generator is evaluated by checking if it **hallucinates** (Faithfulness) and if it **answers the actual question** (Answer Relevancy). Both are measured using an **LLM-as-a-Judge** that breaks down the answer into atomic claims. The best way to improve these scores is to iteratively refine the system prompt based on the reasoning given in failed test cases. 🚀

---

summaries this LLM Evaluation tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples

---

- [Notes](https://onedrive.live.com/personal/85452F67DAA1111C/_layouts/15/Doc.aspx?sourcedoc={90714588-0955-47bc-bf9e-176879959e0d}&action=view&redeem=aHR0cHM6Ly8xZHJ2Lm1zL28vYy84NTQ1MkY2N0RBQTExMTFDL0lnQ0lSWEdRVlFtOFI3LWVGMmg1bFo0TkFheWVYXzlSM1Y0WEhERG1zWFlNbnJr&wd=target%281.%20Introduction%20to%20LLM%20Evals.one%7Ca35dbc27-08ab-4743-b3b0-6b36c29acfd5%2FCourse%20Outline%7C165aacba-2851-e54b-bf9f-885a1a42b9ee%2F%29&wdorigin=NavigationUrl)




