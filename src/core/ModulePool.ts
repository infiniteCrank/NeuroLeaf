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

    evolveModules(): void {
        const clones: Module[] = [];
        const survivors: Module[] = [];

        for (const mod of this.modules) {
            const score = this.computeScore(mod.metrics);
            if (score < 0.3) {
                console.log(`❌ Removing ${mod.id} (score ${score.toFixed(4)})`);
                continue;
            }
            if (score > 0.6) {
                const newId = `${mod.id}-clone-${Date.now()}`;
                console.log(`✨ Cloning ${mod.id} -> ${newId}`);
                const cloned: Module = {
                    ...mod,
                    id: newId,
                };

                // Simple mutation
                if (cloned.elm && cloned.elm.config && typeof cloned.elm.config.dropout === "number") {
                    cloned.elm.config.dropout = Math.max(
                        0,
                        Math.min(
                            0.2,
                            cloned.elm.config.dropout + (Math.random() * 0.02 - 0.01)
                        )
                    );
                }

                clones.push(cloned);
            }
            survivors.push(mod);
        }

        this.modules = [...survivors, ...clones];
    }

    computeScore(metrics: ModuleMetrics): number {
        return (
            metrics.recall1 * 0.5 +
            metrics.recall5 * 0.3 +
            metrics.mrr * 0.2 -
            metrics.avgLatency * 0.01
        );
    }

    /**
     * Each module emits a signal to the bus.
     */
    broadcastAllSignals(bus: SignalBus, vectorizer: TFIDFVectorizer): void {
        for (const mod of this.modules) {
            const text = `Signal from ${mod.id}`;
            const vector = vectorizer.vectorize(text);

            bus.broadcast({
                id: `${mod.id}-${Date.now()}`,
                sourceModuleId: mod.id,
                vector,
                timestamp: Date.now(),
                metadata: {
                    role: mod.role
                }
            });
        }
    }

    /**
     * Each module consumes signals from others.
     */
    consumeAllSignals(bus: SignalBus): void {
        for (const mod of this.modules) {
            const signals = bus.getSignalsFromOthers(mod.id);

            // You can define how to consume them.
            // For example, count how many signals were observed:
            if (signals.length > 0) {
                console.log(`🔊 ${mod.id} observed ${signals.length} signals from peers.`);
            }
        }
    }
}
