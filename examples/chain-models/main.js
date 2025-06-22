// @ts-ignore
const { LanguageClassifier } = window.NeuroLeaf;

window.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const output = document.getElementById('autoOutput');
    const fill = document.getElementById('langFill');

    const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';

    const langConfig = {
        categories: ['English', 'French', 'Spanish'],
        charSet,
        maxLen: 40,
        hiddenUnits: 120,
        activation: 'sigmoid',
        useTokenizer: false
    };

    const hardcodedData = [
        // English
        { text: 'hi', label: 'English' },
        { text: 'hello', label: 'English' },
        { text: 'hey', label: 'English' },
        { text: 'good morning', label: 'English' },
        { text: 'good evening', label: 'English' },
        { text: 'howdy', label: 'English' },
        { text: 'what’s up', label: 'English' },
        { text: 'greetings', label: 'English' },
        { text: 'good day', label: 'English' },
        { text: 'yo', label: 'English' },

        // French
        { text: 'bonjour', label: 'French' },
        { text: 'salut', label: 'French' },
        { text: 'coucou', label: 'French' },
        { text: 'bonsoir', label: 'French' },
        { text: 'bonne journée', label: 'French' },
        { text: 'allô', label: 'French' },
        { text: 'bienvenue', label: 'French' },
        { text: 'bon matin', label: 'French' },
        { text: 'ça va', label: 'French' },
        { text: 'rebonjour', label: 'French' },

        // Spanish
        { text: 'hola', label: 'Spanish' },
        { text: 'buenos días', label: 'Spanish' },
        { text: 'buenas tardes', label: 'Spanish' },
        { text: 'buenas noches', label: 'Spanish' },
        { text: 'qué tal', label: 'Spanish' },
        { text: 'cómo estás', label: 'Spanish' },
        { text: 'saludos', label: 'Spanish' },
        { text: 'ey', label: 'Spanish' },
        { text: 'buen día', label: 'Spanish' },
        { text: 'qué onda', label: 'Spanish' }
    ];

    const lang = new LanguageClassifier(langConfig);
    lang.train(hardcodedData);

    input.addEventListener('input', () => {
        const val = input.value.trim().toLowerCase();
        if (!val) {
            output.textContent = '';
            fill.style.width = '0%';
            fill.textContent = '';
            fill.style.background = '#ccc';
            return;
        }

        output.textContent = `🔍 Testing: "${val}"`;

        const langResults = lang.predict(val);
        console.log('Top language predictions:', langResults);

        const best = langResults[0];
        const percent = Math.round(best.prob * 100);
        fill.style.width = `${percent}%`;
        fill.textContent = `${best.label} (${percent}%)`;

        fill.style.background = {
            English: 'linear-gradient(to right, green, lime)',
            French: 'linear-gradient(to right, blue, cyan)',
            Spanish: 'linear-gradient(to right, red, orange)'
        }[best.label] || '#999';
    });
});
