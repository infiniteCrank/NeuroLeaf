import { ELM } from "./ELM";
import { MiniTransformer } from "./MiniTransformer";

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
}
