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

export class ModulePool {
    modules: Module[];

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

            const fitness = affinity * 0.6 + (usage / (usage + 1)) * 0.4;

            if (fitness < 0.3) {
                console.log(`❌ Removing ${mod.id} (fitness ${fitness.toFixed(2)})`);
                continue;
            }

            if (fitness > 0.7) {
                const newId = `${mod.id}-clone-${Date.now()}`;
                console.log(`✨ Cloning ${mod.id} -> ${newId}`);

                const cloned: Module = {
                    ...mod,
                    id: newId,
                    signalLikes: 0,
                    signalDislikes: 0,
                    signalsConsumedByOthers: 0,
                    emittedSignalIds: [],
                    signalHistory: []
                };
                clones.push(cloned);
            }

            survivors.push(mod);
        }

        this.modules = [...survivors, ...clones];
    }

    /**
     * Each module emits internal, human, and synthetic signals.
     */
    broadcastAllSignals(bus: SignalBus, vectorizer: TFIDFVectorizer): void {
        for (const mod of this.modules) {
            const now = Date.now();

            // 1️⃣ Internal signal
            const internalVector = vectorizer.vectorize(`Embedding from ${mod.id}`);
            const internalSignal: Signal = {
                id: `${mod.id}-internal-${now}`,
                sourceModuleId: mod.id,
                vector: internalVector,
                timestamp: now,
                metadata: {
                    role: "internal"
                }
            };
            bus.broadcast(internalSignal);
            mod.emittedSignalIds.push(internalSignal.id);

            // 2️⃣ Human message signal
            const humanVector = vectorizer.vectorize(`Message from ${mod.id}`);
            const humanSignal: Signal = {
                id: `${mod.id}-human-${now}`,
                sourceModuleId: mod.id,
                vector: humanVector,
                timestamp: now,
                metadata: {
                    role: "human"
                }
            };
            bus.broadcast(humanSignal);
            mod.emittedSignalIds.push(humanSignal.id);

            // 3️⃣ Synthetic signal
            const syntheticVector = Array.from(
                { length: internalVector.length },
                () => Math.random() * 2 - 1
            );
            const syntheticSignal: Signal = {
                id: `${mod.id}-synthetic-${now}`,
                sourceModuleId: mod.id,
                vector: syntheticVector,
                timestamp: now,
                metadata: {
                    role: "synthetic"
                }
            };
            bus.broadcast(syntheticSignal);
            mod.emittedSignalIds.push(syntheticSignal.id);
        }
    }

    /**
     * Each module consumes signals and retrains or mutates based on them.
     */
    consumeAllSignals(bus: SignalBus): void {
        for (const mod of this.modules) {
            const signals = bus.getSignalsFromOthers(mod.id);

            if (signals.length > 0) {
                console.log(`🔊 ${mod.id} observed ${signals.length} signals from peers.`);

                // Keep last 50 signals
                mod.signalHistory.push(...signals);
                if (mod.signalHistory.length > 50) {
                    mod.signalHistory = mod.signalHistory.slice(-50);
                }

                const internalSignals = signals.filter(
                    (s) => s.metadata?.role === "internal"
                );
                mod.signalsConsumedByOthers += internalSignals.length;

                // 🚀 Retrain or mutate if there are internal signals
                if (internalSignals.length > 0 && mod.elm) {
                    const avgVec = this.averageVectors(
                        internalSignals.map((s) => s.vector)
                    );

                    // Dummy label: one-hot with 1 in first position
                    const label = Array(mod.elm.categories.length).fill(0);
                    label[0] = 1;

                    // Retrain
                    mod.elm.trainFromData(
                        [avgVec],
                        [label],
                        { reuseWeights: true }
                    );

                    // Mutate dropout a bit
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

                    console.log(`⚡ ${mod.id} retrained with avg vector from ${internalSignals.length} signals.`);
                }
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
        return sums.map((s) => s / vectors.length);
    }
}
