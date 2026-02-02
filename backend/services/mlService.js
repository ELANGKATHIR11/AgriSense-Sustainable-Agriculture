const { spawn } = require('child_process');
const path = require('path');
const { pythonPath } = require('../config/pythonConfig');

class MLService {
    constructor() {
        this.pythonPath = pythonPath;
        this.scriptPath = path.join(__dirname, '../ml/unified_inference.py');
    }

    async analyze(inputData) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn(this.pythonPath, [this.scriptPath]);
            
            let dataString = '';
            let errorString = '';

            pythonProcess.stdin.write(JSON.stringify(inputData));
            pythonProcess.stdin.end();

            pythonProcess.stdout.on('data', (data) => {
                dataString += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorString += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(dataString);
                        if (result.error) reject(new Error(result.error));
                        resolve(result);
                    } catch (e) {
                        reject(new Error(`Failed to parse ML output: ${dataString}`));
                    }
                } else {
                    reject(new Error(`ML Process exited with code ${code}: ${errorString}`));
                }
            });

            // Timeout after 15 seconds
            setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('ML Inference Timeout'));
            }, 15000);
        });
    }
}

module.exports = new MLService();
