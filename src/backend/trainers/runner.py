import os
import subprocess
import argparse
import sys

def run_existing_script(script_path, extra_args):
    cmd = [sys.executable, script_path] + extra_args
    return subprocess.call(cmd)

def tiny_tf_smoke_train(epochs, batch_size):
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    x = np.random.randn(256, 32).astype('float32')
    y = np.random.randint(0, 2, size=(256, 1)).astype('float32')

    model = keras.Sequential([
        keras.layers.Input(shape=(32,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x, y, epochs=epochs, batch_size=batch_size)
    print('Tiny TF smoke training complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['llm','vlm','hybrid'], required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Look for user-provided training entrypoints first
    project_scripts = {
        'llm': ['train_llm.py', 'agrisense_app/train_llm.py', 'train/llm.py'],
        'vlm': ['train_vlm.py', 'agrisense_app/train_vlm.py', 'train/vlm.py'],
        'hybrid': ['train_hybrid.py', 'agrisense_app/train_hybrid.py', 'train/hybrid.py']
    }

    for candidate in project_scripts[args.model]:
        if os.path.exists(candidate):
            print(f'Found project script {candidate}; invoking it')
            rc = run_existing_script(candidate, args.extra)
            sys.exit(rc)

    # Fallback: run tiny TF smoke training
    print('No project training script found. Running tiny TF smoke training as fallback.')
    tiny_tf_smoke_train(args.epochs, args.batch_size)


if __name__ == '__main__':
    main()
