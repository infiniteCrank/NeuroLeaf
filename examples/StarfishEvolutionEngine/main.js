const {
    ModulePool,
    ELM,
    SignalBus,
    TFIDFVectorizer
} = window.astermind;

const pool = new ModulePool();
const bus = new SignalBus();
const defaultCorpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Cybernetic organisms adapt and learn from signals.",
    "Extreme Learning Machines are efficient and powerful.",
    "Signal buses distribute input across connected modules."
];

const vectorizer = new TFIDFVectorizer(defaultCorpus);

const canvas = document.getElementById("cyberCanvas");
const ctx = canvas.getContext("2d");

function drawModuleActivity(modules, timestampMap) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    modules.forEach((mod, index) => {
        const x = 40 * (index % 10) + 10;
        const y = 40 * Math.floor(index / 10) + 10;

        const active = timestampMap[mod.id] && Date.now() - timestampMap[mod.id] < 3000;
        ctx.fillStyle = active ? "#00ffcc" : "#333";
        ctx.fillRect(x, y, 20, 20);

        ctx.strokeStyle = "#ff00ff";
        ctx.strokeRect(x, y, 20, 20);
    });
}

const timestampMap = {}; // tracks last active time

// Create 10 ELM modules
for (let i = 0; i < 10; i++) {
    const elm = new ELM({
        categories: ["default"],
        hiddenUnits: 20,
        maxLen: 40,
        activation: "relu",
        dropout: 0.1,
        charSet: "abcdefghijklmnopqrstuvwxyz ",
        useTokenizer: false,
        log: { verbose: false }
    });

    pool.addModule({
        id: `mod-${i}`,
        type: "ELM",
        elm,
        role: "listener",
        metrics: {
            recall1: 0,
            recall5: 0,
            mrr: 0,
            avgLatency: 0,
        },
        lastEvaluated: Date.now()
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("userInput");

    input.addEventListener("input", () => {
        const typed = input.value.trim();
        if (!typed) return;

        pool.listModules().forEach(mod => {
            if (mod.elm) {
                mod.elm.setCategories([typed]);
                mod.elm.train({ includeNoise: true });

                // mark active
                timestampMap[mod.id] = Date.now();
            }
        });

        pool.broadcastAllSignals(bus, vectorizer);
        pool.consumeAllSignals(bus);

        drawModuleActivity(pool.listModules(), timestampMap);
    });

    // Evolution loop every 5 seconds
    setInterval(() => {
        //pool.evolveModules();
        drawModuleActivity(pool.listModules(), timestampMap);
    }, 5000);
});
