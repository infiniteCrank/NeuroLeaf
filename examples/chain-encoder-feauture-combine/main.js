// @ts-ignore
const { AutoComplete, EncoderELM, FeatureCombinerELM, RefinerELM } = window.NeuroLeaf;

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
    const maxLen = 30;

    const acConfig = {
        charSet,
        maxLen,
        hiddenUnits: 40,
        activation: 'relu',
        useTokenizer: true
    };

    const encoderConfig = {
        charSet,
        maxLen,
        hiddenUnits: 32,
        activation: 'relu',
        useTokenizer: true
    };

    const combinerConfig = {
        charSet,
        maxLen,
        hiddenUnits: 64,
        activation: 'relu',
        useTokenizer: false
    };

    const refinerConfig = {
        charSet,
        maxLen,
        hiddenUnits: 64,
        activation: 'relu',
        useTokenizer: false
    };

    fetch('/language_greetings_1500.csv')
        .then(r => r.text())
        .then(csv => {
            const rows = csv.split('\n').map(l => l.trim()).filter(Boolean).slice(1);
            const allData = rows.map(l => {
                const [text = '', label = ''] = l.split(',');
                return { text: text.trim().toLowerCase(), label: label.trim() };
            });

            const greetings = allData.map(d => d.text);
            const labels = allData.map(d => d.label);

            const ac = new AutoComplete([...new Set(greetings)], {
                ...acConfig,
                inputElement: input,
                outputElement: output
            });

            const encoder = new EncoderELM(encoderConfig);
            const vectors = greetings.map(g => encoder.elm.encoder.normalize(encoder.elm.encoder.encode(g)));
            encoder.train(greetings, vectors);

            const metas = greetings.map(g => [
                g.length,
                new Set(g).size / g.length
            ]);

            const combiner = new FeatureCombinerELM(combinerConfig);
            combiner.train(vectors, metas, labels);

            const combinedInputs = vectors.map((vec, i) =>
                FeatureCombinerELM.combineFeatures(vec, metas[i])
            );

            // Evaluate combiner confidence
            const combinerResults = vectors.map((vec, i) =>
                combiner.predict(vec, metas[i])[0]
            );

            const lowConfidence = combinerResults
                .map((res, i) => ({
                    vector: combinedInputs[i],
                    label: labels[i],
                    prob: res.prob
                }))
                .filter(d => d.prob < 0.6);

            const refiner = new RefinerELM(refinerConfig);

            if (lowConfidence.length > 0) {
                refiner.train(
                    lowConfidence.map(d => d.vector),
                    lowConfidence.map(d => d.label)
                );
                console.log(`🔁 Refiner trained on ${lowConfidence.length} low-confidence samples.`);
            } else {
                console.log('⚠️ No low-confidence samples to train the Refiner.');
            }

            input.addEventListener('input', () => {
                const val = input.value.trim().toLowerCase();
                if (!val) {
                    output.textContent = '';
                    fill.style.width = '0%';
                    fill.textContent = '';
                    fill.style.background = '#ccc';
                    return;
                }

                const [acResult] = ac.predict(val, 1);
                const prediction = acResult.completion;
                output.textContent = `🔮 Autocomplete: ${prediction}`;

                const encoded = encoder.encode(prediction);
                const meta = [
                    val.length,
                    new Set(val).size / val.length
                ];

                const combinedVec = FeatureCombinerELM.combineFeatures(encoded, meta);
                const [combinerResult] = combiner.predict(encoded, meta);

                let finalResult = combinerResult;

                if (combinerResult.prob < 0.6) {
                    try {
                        const [refined] = refiner.predict(combinedVec);
                        if (refined) {
                            finalResult = refined;
                            console.log('🔁 Used Refiner');
                        }
                    } catch (err) {
                        console.warn('⚠️ Refiner not available or failed:', err);
                    }
                }

                console.log('🧪 Combiner:', combinerResult);
                console.log('🧪 Refiner:', refiner.predict(combinedVec)[0]);

                const percent = Math.round(finalResult.prob * 100);
                fill.style.width = `${percent}%`;

                fill.textContent = `${finalResult.label} (${percent}%)`;
                fill.dataset.model = finalResult === combinerResult ? 'combiner' : 'refiner';
                fill.textContent = `${finalResult.label} (${percent}%)` + (finalResult === combinerResult ? ' [C]' : ' [R]');

                fill.style.background = {
                    English: 'linear-gradient(to right, green, lime)',
                    French: 'linear-gradient(to right, blue, cyan)',
                    Spanish: 'linear-gradient(to right, red, orange)'
                }[finalResult.label] || '#999';
            });
        });
});
