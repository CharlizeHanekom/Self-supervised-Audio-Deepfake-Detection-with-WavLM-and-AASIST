import argparse

def get_args():
    """
    Defines and parses command-line arguments for training and evaluation.
    """
    parser = argparse.ArgumentParser(description="WavLM-AASIST Audio Deepfake Detection")

    # Common arguments for both training and evaluation
    parser.add_argument('--data_dir', type=str, default='./data',help='Root directory of datasets')
    parser.add_argument('--model_path', type=str, default='./models/wavlm-base',help='Path to pretrained WavLM model')
    parser.add_argument('--output_dir', type=str, default='./output',help='Directory to save model checkpoints and results')
    parser.add_argument('--batch_size', type=int, default=10,help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=8,help='Number of workers for data loading')
    parser.add_argument('--model_checkpoint', type=str, default='best_model.pth',help='Filename for the saved model checkpoint')

    # Arguments specific to training
    parser.add_argument('--epochs', type=int, default=50,help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5,help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=1e-4,help='Initial learning rate for the backend/classifier')

    # Arguments for fine-tuning
    parser.add_argument('--fine_tune', action='store_true',help='If set, the WavLM model will be fine-tuned')
    parser.add_argument('--wavlm_lr_ratio', type=float, default=0.1,help='Ratio of WavLM learning rate to the main learning rate during fine-tuning')

    return parser.parse_args()


#COMMANDS TO RUN
# pip install -r requirements.txt

## To train normally
# python train.py --epochs 50 --patience 5 --batch_size 10

## To train with fine-tuning (recommended for better results)
# python train.py --epochs 15 --patience 3 --batch_size 8 --fine_tune

## To evaluate a trained model
# python evaluate.py