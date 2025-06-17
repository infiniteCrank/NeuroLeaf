import typescript from 'rollup-plugin-typescript2';

export default {
    input: 'index.ts',
    output: [
        {
            file: 'dist/neuroleaf.cjs.js',
            format: 'cjs', // CommonJS for Node.js
            sourcemap: true
        },
        {
            file: 'dist/neuroleaf.esm.js',
            format: 'esm', // For bundlers like Webpack or native import
            sourcemap: true
        },
        {
            file: 'dist/neuroleaf.umd.js',
            format: 'umd', // Universal for browser
            name: 'neuroleaf',
            sourcemap: true
        }
    ],
    plugins: [typescript()]
};
