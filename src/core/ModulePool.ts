import { TFIDFVectorizer } from "../ml/TFIDF";
import { ELM } from "./ELM";
import { MiniTransformer } from "./MiniTransformer";
import { Signal, SignalBus } from "./SignalBus";

export type ModuleType = "ELM" | "Transformer";

export interface ModuleMetrics {
    recall1: number;
    recall5: number;
    mrr: number;
    avgLatency: number;
}

export interface Module {
    id: string;
    type: ModuleType;
    elm?: ELM;
    transformer?: MiniTransformer;
    role: string;
    metrics: ModuleMetrics;
    lastEvaluated: number;
    signalLikes: number;
    signalDislikes: number;
    signalsConsumedByOthers: number;
    emittedSignalIds: string[];
    signalHistory: Signal[];
    lastNovelty?: number;
}

export interface Feedback {
    signalId: string;
    moduleId: string;
    score: 1 | -1;
}

export class FeedbackStore {
    private feedbacks: Feedback[] = [];

    addFeedback(f: Feedback): void {
        this.feedbacks.push(f);
    }

    getModuleFeedback(moduleId: string): { likes: number; dislikes: number } {
        let likes = 0, dislikes = 0;
        for (const f of this.feedbacks) {
            if (f.moduleId === moduleId) {
                if (f.score > 0) likes++;
                else dislikes++;
            }
        }
        return { likes, dislikes };
    }
}

const MODULE_ROLES = ["listener", "communicator", "explorer", "synthesizer"];

export class ModulePool {
    modules: Module[];
    private moduleFitness: Map<string, number> = new Map();

    constructor() {
        this.modules = [];
    }

    addModule(mod: Module) {
        this.modules.push(mod);
    }

    removeModule(id: string) {
        this.modules = this.modules.filter((m) => m.id !== id);
    }

    getModuleById(id: string): Module | undefined {
        return this.modules.find((m) => m.id === id);
    }

    listModules(): Module[] {
        return this.modules;
    }

    evolveModules(feedbackStore: FeedbackStore): void {
        const clones: Module[] = [];
        const survivors: Module[] = [];

        for (const mod of this.modules) {
            const { likes, dislikes } = feedbackStore.getModuleFeedback(mod.id);
            const usage = mod.signalsConsumedByOthers;
            const totalFeedback = likes + dislikes || 1;
            const affinity = likes / totalFeedback;

            const roleBoost = mod.role === "communicator" ? 0.1 : 0;
            const noveltyBoost = (mod.lastNovelty ?? 0) * 0.2;

            const fitness = affinity * 0.5 + (usage / (usage + 1)) * 0.3 + roleBoost + noveltyBoost;
            this.moduleFitness.set(mod.id, fitness);

            if (fitness < 0.3) {
                console.log(`❌ Removing ${mod.id} (fitness ${fitness.toFixed(2)})`);
                continue;
            }

            if (fitness > 0.7) {
                const newId = `${mod.id}-clone-${Date.now()}`;
                const newRole = Math.random() < 0.3
                    ? MODULE_ROLES[Math.floor(Math.random() * MODULE_ROLES.length)]
                    : mod.role;

                if (newRole !== mod.role) {
                    console.log(`🔄 ${mod.id} changed role: ${mod.role} ➡️ ${newRole}`);
                }

                console.log(`✨ Cloning ${mod.id} -> ${newId}`);

                const cloned: Module = {
                    ...mod,
                    id: newId,
                    role: newRole,
                    signalLikes: 0,
                    signalDislikes: 0,
                    signalsConsumedByOthers: 0,
                    emittedSignalIds: [],
                    signalHistory: [],
                    lastNovelty: undefined
                };
                clones.push(cloned);
            }

            survivors.push(mod);
        }

        this.modules = [...survivors, ...clones];
    }

    broadcastAllSignals(bus: SignalBus, vectorizer: TFIDFVectorizer): void {
        for (const mod of this.modules) {
            const now = Date.now();

            const internalId = `${mod.id}-internal-${now}-${Math.floor(Math.random() * 1000)}`;
            const internalVector = vectorizer.vectorize(`Embedding from ${mod.id}`);
            bus.broadcast({
                id: internalId,
                sourceModuleId: mod.id,
                vector: internalVector,
                timestamp: now,
                metadata: { role: "internal", parents: [] }
            });
            mod.emittedSignalIds.push(internalId);

            const humanId = `${mod.id}-human-${now}-${Math.floor(Math.random() * 1000)}`;
            const humanVector = vectorizer.vectorize(`Message from ${mod.id}`);
            bus.broadcast({
                id: humanId,
                sourceModuleId: mod.id,
                vector: humanVector,
                timestamp: now,
                metadata: { role: "human", parents: [] }
            });
            mod.emittedSignalIds.push(humanId);

            if (Math.random() < 0.3) {
                let syntheticVector: number[];
                let parentIds: string[] = [];

                if (Math.random() < 0.5) {
                    syntheticVector = Array.from(
                        { length: internalVector.length },
                        () => Math.random() * 2 - 1
                    );
                    console.log(`🌱 ${mod.id} emitted RANDOM synthetic signal.`);
                } else if (mod.signalHistory.length > 0) {
                    const signalsToCombine = mod.signalHistory.slice(-5);
                    syntheticVector = this.averageVectors(signalsToCombine.map(s => s.vector));
                    parentIds = signalsToCombine.map(s => s.id);
                    console.log(`🌿 ${mod.id} emitted RECOMBINED synthetic signal from ${parentIds.length} parents.`);
                } else {
                    syntheticVector = Array.from(
                        { length: internalVector.length },
                        () => Math.random() * 2 - 1
                    );
                    console.log(`🌱 ${mod.id} emitted fallback RANDOM synthetic signal.`);
                }

                const syntheticId = `${mod.id}-synthetic-${now}-${Math.floor(Math.random() * 1000)}`;
                bus.broadcast({
                    id: syntheticId,
                    sourceModuleId: mod.id,
                    vector: syntheticVector,
                    timestamp: now,
                    metadata: {
                        role: "synthetic",
                        parents: parentIds
                    }
                });
                mod.emittedSignalIds.push(syntheticId);
            }
        }
    }

    getSignalLineage(signalId: string, bus: SignalBus): string[] {
        const signal = bus.getSignals().find(s => s.id === signalId);
        if (!signal) return [];

        const parents = signal.metadata?.parents ?? [];
        let lineage = [...parents];

        for (const parentId of parents) {
            const ancestorLine = this.getSignalLineage(parentId, bus);
            lineage = lineage.concat(ancestorLine);
        }

        return lineage;
    }

    private weightedAverageVectors(signals: Signal[], bus: SignalBus): number[] {
        if (signals.length === 0) return [];

        const length = signals[0].vector.length;
        const sums = Array(length).fill(0);
        let totalWeight = 0;

        for (const signal of signals) {
            const parentModuleId = signal.sourceModuleId;
            const medianFitness = this.getMedianFitness();
            const fitness = this.moduleFitness.get(parentModuleId) ?? medianFitness;
            const novelty = this.computeSignalNovelty(signal, bus);
            const weight = 0.1 + 0.7 * fitness + 0.3 * novelty;

            for (let i = 0; i < length; i++) {
                sums[i] += signal.vector[i] * weight;
            }

            totalWeight += weight;
        }

        if (totalWeight === 0) return Array(length).fill(0);

        return sums.map(s => s / totalWeight);
    }

    private getMedianFitness(): number {
        const fitnesses = Array.from(this.moduleFitness.values());
        if (fitnesses.length === 0) return 0.5;

        fitnesses.sort((a, b) => a - b);
        const mid = Math.floor(fitnesses.length / 2);

        return fitnesses.length % 2 !== 0
            ? fitnesses[mid]
            : (fitnesses[mid - 1] + fitnesses[mid]) / 2;
    }

    consumeAllSignals(bus: SignalBus): void {
        for (const mod of this.modules) {
            const signals = bus.getSignalsFromOthers(mod.id);
            if (signals.length === 0) continue;

            console.log(`🔊 ${mod.id} observed ${signals.length} signals.`);

            mod.signalHistory.push(...signals);
            if (mod.signalHistory.length > 50) {
                mod.signalHistory = mod.signalHistory.slice(-50);
            }

            const internalSignals = signals.filter(s => s.metadata?.role === "internal");
            mod.signalsConsumedByOthers += internalSignals.length;

            if (internalSignals.length > 0 && mod.elm) {
                const avgVec = this.weightedAverageVectors(internalSignals, bus);
                const novelty = this.computeNoveltyScore(avgVec, mod.signalHistory);

                mod.lastNovelty = Math.min(novelty, 1);

                const hash = "cat-" + Math.abs(Math.floor(avgVec.reduce((a, b) => a + b, 0) * 1000));
                if (!mod.elm.categories.includes(hash)) {
                    mod.elm.categories.push(hash);
                    console.log(`🪴 ${mod.id} discovered new category: ${hash}`);
                }
                const catIndex = mod.elm.categories.indexOf(hash);
                const label = Array(mod.elm.categories.length).fill(0);
                label[catIndex] = 1;

                mod.elm.trainFromData(
                    [avgVec],
                    [label],
                    { reuseWeights: true }
                );

                if (typeof mod.elm.config.dropout === "number") {
                    const oldDropout = mod.elm.config.dropout;
                    mod.elm.config.dropout = Math.max(
                        0,
                        Math.min(
                            0.2,
                            oldDropout + (Math.random() * 0.02 - 0.01)
                        )
                    );
                    console.log(`⚙️ ${mod.id} adjusted dropout: ${oldDropout.toFixed(3)} -> ${mod.elm.config.dropout.toFixed(3)}`);
                }

                console.log(`⚡ ${mod.id} retrained with novelty=${novelty.toFixed(3)} (fitness-weighted ancestors)`);
            }
        }
    }

    private averageVectors(vectors: number[][]): number[] {
        if (vectors.length === 0) return [];
        const length = vectors[0].length;
        const sums = Array(length).fill(0);
        for (const vec of vectors) {
            for (let i = 0; i < length; i++) {
                sums[i] += vec[i];
            }
        }
        return sums.map(s => s / vectors.length);
    }

    private computeNoveltyScore(vector: number[], history: Signal[]): number {
        if (history.length === 0) return 1;
        const lastVectors = history.map(s => s.vector);
        const avgVec = this.averageVectors(lastVectors);

        const dot = vector.reduce((sum, v, i) => sum + v * avgVec[i], 0);
        const normA = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
        const normB = Math.sqrt(avgVec.reduce((sum, v) => sum + v * v, 0));
        const similarity = dot / (normA * normB + 1e-8);
        return 1 - similarity;
    }

    private findSignalById(id: string, bus: SignalBus): Signal | undefined {
        return bus.getSignals().find(s => s.id === id);
    }

    private getAncestorVectors(signal: Signal, bus: SignalBus): number[][] {
        const vectors: number[][] = [];

        if (!signal.metadata?.parents) return vectors;

        for (const parentId of signal.metadata.parents) {
            const parent = this.findSignalById(parentId, bus);
            if (parent?.vector) {
                vectors.push(parent.vector);
                const further = this.getAncestorVectors(parent, bus);
                vectors.push(...further);
            }
        }

        return vectors;
    }

    private computeSignalNovelty(signal: Signal, bus: SignalBus): number {
        if (!signal.vector || signal.vector.length === 0) return 0;
        if (!signal.metadata?.parents || signal.metadata.parents.length === 0) return 1;

        const parentVectors: number[][] = [];

        for (const parentId of signal.metadata.parents) {
            const parent = this.findSignalById(parentId, bus);
            if (parent?.vector) {
                parentVectors.push(parent.vector);
                const grandParents = this.getAncestorVectors(parent, bus);
                parentVectors.push(...grandParents);
            }
        }

        if (parentVectors.length === 0) return 1;

        const avgParent = this.averageVectors(parentVectors);
        return this.computeNoveltyScore(signal.vector, [{ vector: avgParent } as Signal]);
    }
}
