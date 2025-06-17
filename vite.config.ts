import { defineConfig } from 'vite';

export default defineConfig(({ command }) => {
    const isDemo = command === 'serve' && process.cwd().includes('examples');
    const isVitest = !!process.env.VITEST; // ✅ safest way to detect Vitest

    return {
        root: isDemo ? 'examples/music-autocomplete-demo' : '.',
        publicDir: isDemo ? '../../public' : 'public',
        server: {
            port: 5173,
        },
        test: {
            include: ['tests/**/*.test.ts'],
            environment: isVitest ? 'jsdom' : 'node', // ✅ avoid comparing against invalid command
        },
    };
});
