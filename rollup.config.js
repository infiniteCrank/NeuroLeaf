import typescript from 'rollup-plugin-typescript2';

export default {
    input: 'index.ts',
    output: [
        {
            file: 'dist/elm-lib.cjs.js',
            format: 'cjs', // CommonJS for Node.js
            sourcemap: true
        },
        {
            file: 'dist/elm-lib.esm.js',
            format: 'esm', // For bundlers like Webpack or native import
            sourcemap: true
        },
        {
            file: 'dist/elm-lib.umd.js',
            format: 'umd', // Universal for browser
            name: 'ELMLib',
            sourcemap: true
        }
    ],
    plugins: [typescript()]
};
