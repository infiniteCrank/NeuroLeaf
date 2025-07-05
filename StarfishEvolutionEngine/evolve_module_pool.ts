// evolve_module_pool.ts
import { ModulePool, Module } from "../src/core/ModulePool";
import { ELM } from "../src/core/ELM";
import { MiniTransformer, Vocab } from "../src/core/MiniTransformer";
import fs from "fs";

// Load evaluated modules
const pool = new ModulePool();
const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
loaded.forEach((m: Module) => pool.addModule(m));

// Rank modules by fitness
function fitness(m: Module) {
    const r1 = m.metrics?.recall1 ?? 0;
    const latency = m.metrics?.avgLatency ?? 1000;
    // Lower latency = higher fitness
    return r1 - latency / 5000;
}

// Sort descending
const ranked = pool.listModules().sort((a, b) => fitness(b) - fitness(a));

// Keep top 5
const survivors = ranked.slice(0, 5);

// Generate new mutated modules to refill pool to size 10
const newModules: Module[] = [];
for (let i = 0; i < 5; i++) {
    const parent = survivors[i % survivors.length];

    if (parent.elm) {
        // Mutate ELM
        const dropout = Math.max(0, Math.min(0.2, (parent.elm.config.dropout ?? 0.02) + (Math.random() * 0.02 - 0.01)));
        const hiddenUnits = Math.max(8, parent.elm.hiddenUnits + Math.floor(Math.random() * 16 - 8));
        const activation = ["relu", "tanh", "leakyRelu"][Math.floor(Math.random() * 3)];
        const child = new ELM({
            ...parent.elm.config,
            hiddenUnits,
            activation,
            dropout
        });

        newModules.push({
            id: `elm-${Date.now()}-${i}`,
            type: "ELM",
            elm: child,
            role: "retrieval",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now()
        });
    }

    if (parent.transformer) {
        // Mutate Transformer
        const embedDim = Math.max(4, parent.transformer.embedDim + Math.floor(Math.random() * 4 - 2));
        const seqLen = Math.max(4, parent.transformer.seqLen + Math.floor(Math.random() * 2 - 1));
        const child = new MiniTransformer(parent.transformer.vocab);
        child.embedDim = embedDim;
        child.seqLen = seqLen;
        child.embedding = child.randomMatrix(parent.transformer.vocab.tokens.length, embedDim);
        child.posEnc = child.positionalEncoding(seqLen, embedDim);

        newModules.push({
            id: `transformer-${Date.now()}-${i}`,
            type: "Transformer",
            transformer: child,
            role: "retrieval",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now()
        });
    }
}

// Reset pool: keep survivors + add new modules
const evolvedPool = new ModulePool();
[...survivors, ...newModules].forEach(m => evolvedPool.addModule(m));

// Save
fs.writeFileSync("./module_pool.json", JSON.stringify(evolvedPool.listModules(), null, 2));
console.log(`✅ Evolved module pool. Kept ${survivors.length}, added ${newModules.length}.`);
