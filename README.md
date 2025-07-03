# 🌟 AsterMind: Decentralized ELM Framework Inspired by Nature

Welcome to **AsterMind**, a modular, decentralized machine learning framework built around small, cooperating Extreme Learning Machines (ELMs) that self-train, self-evaluate, and self-repair—just like the decentralized nervous system of a starfish.

AsterMind is designed for:

* Lightweight, in-browser ML pipelines
* Transparent, interpretable predictions
* Continuous, incremental learning
* Resilient systems with no single point of failure

---

## ✨ Features

✅ Modular Architecture
✅ Self-Governing Training
✅ Flexible Preprocessing
✅ Lightweight Deployment
✅ Retrieval and Classification Utilities

---

## 🚀 Installation

Clone the repository and import modules:

```bash
https://github.com/infiniteCrank/astermind
```

---

## 🛠️ Usage Example

Define config, initialize an ELM, load or train model, predict:

```typescript
const config = { categories: ['English', 'French'], hiddenUnits: 128 };
const elm = new ELM(config);
// Load or train logic here
const results = elm.predict("bonjour");
```

---

## 🧪 Suggested Experiments

* Compare retrieval performance with Sentence-BERT and TFIDF.
* Experiment with activations and token vs char encoding.
* Deploy in-browser retraining workflows.

---

## 🌿 Why Use AsterMind?

Because you can build AI systems that:

* Are decentralized.
* Self-heal and retrain independently.
* Run in the browser.
* Are transparent and interpretable.

---
## 📚 Core API Documentation

### ELM Class

**Constructor:**

```typescript
new ELM(config: ELMConfig)
```

* `config`: Configuration object specifying categories, hidden units, activation, metrics, and more.

**Methods:**

* `train(augmentationOptions?, weights?)`: Trains the model using auto-generated training data.
* `trainFromData(X, Y, options?)`: Trains the model using provided matrices.
* `predict(text, topK)`: Predicts probabilities for each label.
* `predictFromVector(vector, topK)`: Predicts from a pre-encoded input.
* `loadModelFromJSON(json)`: Loads a model from saved JSON.
* `saveModelAsJSONFile(filename?)`: Saves the model to disk.
* `computeHiddenLayer(X)`: Computes hidden layer activations.
* `getEmbedding(X)`: Returns embeddings.
* `calculateRMSE`, `calculateMAE`, `calculateAccuracy`, `calculateF1Score`, `calculateCrossEntropy`, `calculateR2Score`: Evaluation metrics.

### ELMChain Class

**Constructor:**

```typescript
new ELMChain(encoders: ELM[])
```

**Methods:**

* `getEmbedding(X)`: Sequentially passes data through all encoders.

### TFIDFVectorizer Class

* `vectorize(doc)`: Converts text into TFIDF vector.
* `vectorizeAll()`: Converts all training documents.

### KNN

* `KNN.find(queryVec, dataset, k, topX, metric)`: Finds k nearest neighbors.

For detailed examples, see `examples/` folder in the repository.

## 📚 Core API Documentation with Examples

### ELM Class

**Constructor:**

```typescript
const elm = new ELM({
  categories: ["English", "French"],
  hiddenUnits: 100,
  activation: "relu",
  log: { modelName: "LangModel" }
});
```

**Example Training:**

```typescript
elm.train();
```

**Example Prediction:**

```typescript
const results = elm.predict("bonjour");
console.log(results);
```

**Diagram:**

```
Input Text -> UniversalEncoder -> Hidden Layer -> Output Weights -> Probabilities
```

---

### ELMChain Class

**Constructor:**

```typescript
const chain = new ELMChain([encoderELM, classifierELM]);
```

**Embedding Example:**

```typescript
const embedding = chain.getEmbedding([vector]);
```

**Diagram:**

```
Input -> ELM1 -> Embedding -> ELM2 -> Final Embedding
```

---

### TFIDFVectorizer Class

**Example:**

```typescript
const vectorizer = new TFIDFVectorizer(["text one", "text two"]);
const vector = vectorizer.vectorize("text one");
```

**Diagram:**

```
Text -> Tokenization -> TFIDF Vector
```

---

### KNN

**Example:**

```typescript
const neighbors = KNN.find(queryVec, dataset, 5, 3, "cosine");
```

**Diagram:**

```
Query Vector -> Similarity -> Nearest Neighbors
```

---

For more examples, see the `examples/` folder.

## 📄 License

MIT License

---

**"AsterMind doesn’t just mimic a brain—it functions more like a starfish: fully decentralized, self-evaluating, and self-repairing."**
