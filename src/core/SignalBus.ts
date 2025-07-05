// src/core/SignalBus.ts

export interface Signal {
    id: string;
    sourceModuleId: string;
    vector: number[];
    timestamp: number;
    metadata?: Record<string, any>;
}

export class SignalBus {
    private signals: Signal[] = [];
    private maxHistory: number;

    constructor(maxHistory = 500) {
        this.maxHistory = maxHistory;
    }

    // Broadcast a new signal
    broadcast(signal: Signal): void {
        this.signals.push(signal);
        if (this.signals.length > this.maxHistory) {
            this.signals.shift(); // remove oldest
        }
    }

    // Retrieve all signals
    getSignals(): Signal[] {
        return this.signals.slice();
    }

    // Retrieve signals from other modules only
    getSignalsFromOthers(moduleId: string): Signal[] {
        return this.signals.filter(s => s.sourceModuleId !== moduleId);
    }

    // Clear all signals (optional)
    clear(): void {
        this.signals = [];
    }
}
