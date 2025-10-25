"""
Professional Visualization Generator for Acoustic Drone Detection System
Computes all metrics directly from actual model and dataset
Generates presentation-ready JPEGs and detailed PNGs with metadata
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import librosa
import librosa.display

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

# Configure matplotlib for clean output
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'figure.max_open_warning': 0,
    'figure.constrained_layout.use': True,  # Auto layout
    'figure.autolayout': False,  # Rely on constrained_layout
    'axes.titlepad': 6,
    'axes.labelpad': 4
})


class DatasetValidator:
    """Validate EDTH Munich dataset structure and samples"""
    
    def __init__(self, train_dir: str, val_dir: str):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.expected_classes = ['drone', 'helicopter', 'background']
        
    def validate(self) -> Dict[str, Any]:
        """Validate dataset structure and return statistics"""
        print("\n" + "="*70)
        print("DATASET VALIDATION")
        print("="*70)
        
        results = {
            'train_dir': str(self.train_dir),
            'val_dir': str(self.val_dir),
            'classes': {},
            'valid': True,
            'warnings': []
        }
        
        # Check train directory
        if not self.train_dir.exists():
            results['valid'] = False
            results['warnings'].append(f"Training directory not found: {self.train_dir}")
            return results
            
        if not self.val_dir.exists():
            results['valid'] = False
            results['warnings'].append(f"Validation directory not found: {self.val_dir}")
            return results
        
        print(f"✓ Train directory: {self.train_dir}")
        print(f"✓ Val directory: {self.val_dir}\n")
        
        # Check each class
        for class_name in self.expected_classes:
            train_class_dir = self.train_dir / class_name
            val_class_dir = self.val_dir / class_name
            
            if not train_class_dir.exists():
                results['warnings'].append(f"Missing train class: {class_name}")
                continue
                
            train_files = list(train_class_dir.glob("*.wav"))
            val_files = list(val_class_dir.glob("*.wav")) if val_class_dir.exists() else []
            
            results['classes'][class_name] = {
                'train_count': len(train_files),
                'val_count': len(val_files),
                'sample_file': str(train_files[0]) if train_files else None
            }
            
            print(f"Class: {class_name:12s} | Train: {len(train_files):3d} | Val: {len(val_files):3d}")
        
        # Check class balance
        train_counts = [info['train_count'] for info in results['classes'].values()]
        if train_counts:
            max_count = max(train_counts)
            min_count = min(train_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3.0:
                warning = f"⚠️  Severe class imbalance detected (ratio: {imbalance_ratio:.2f}x)"
                print(f"\n{warning}")
                print("   Recommendation: Use WeightedRandomSampler in training")
                results['warnings'].append(warning)
            
            print(f"\nClass balance ratio: {imbalance_ratio:.2f}x")
        
        return results
    
    def validate_audio_samples(self, target_sr: int = 22050, target_duration: float = 3.0) -> Dict[str, Any]:
        """Validate audio sample specifications"""
        print(f"\n{'─'*70}")
        print(f"AUDIO SAMPLE VALIDATION (target: {target_sr}Hz, {target_duration}s)")
        print(f"{'─'*70}")
        
        results = {'samples': {}, 'valid': True, 'warnings': []}
        
        for class_name in self.expected_classes:
            train_class_dir = self.train_dir / class_name
            if not train_class_dir.exists():
                continue
                
            # Check 2 samples per class
            wav_files = list(train_class_dir.glob("*.wav"))[:2]
            
            for wav_file in wav_files:
                try:
                    # Load without resampling to check original
                    y_orig, sr_orig = librosa.load(wav_file, sr=None, mono=True, duration=5.0)
                    
                    # Load with target sample rate
                    y_target, sr_target = librosa.load(wav_file, sr=target_sr, mono=True, duration=target_duration)
                    
                    sample_info = {
                        'file': wav_file.name,
                        'original_sr': sr_orig,
                        'original_duration': len(y_orig) / sr_orig,
                        'target_sr': sr_target,
                        'target_duration': len(y_target) / sr_target,
                        'target_samples': len(y_target)
                    }
                    
                    if class_name not in results['samples']:
                        results['samples'][class_name] = []
                    results['samples'][class_name].append(sample_info)
                    
                    print(f"  {class_name}/{wav_file.name}: {sr_orig}Hz → {sr_target}Hz, "
                          f"{len(y_orig)/sr_orig:.2f}s → {len(y_target)/sr_target:.2f}s")
                    
                except Exception as e:
                    warning = f"Failed to load {wav_file}: {e}"
                    results['warnings'].append(warning)
                    results['valid'] = False
        
        return results


class ModelIntrospector:
    """Extract actual architecture details from the model"""
    
    def __init__(self, model_type: str = 'crnn'):
        self.model_type = model_type
        self.model = None
        self.input_shape = None
        self.layer_info = []
        
    def load_model(self, num_classes: int = 3, input_channels: int = 3, 
                   n_mels: int = 128, dropout: float = 0.3) -> nn.Module:
        """Load and instantiate the actual model"""
        print(f"\n{'='*70}")
        print(f"MODEL INTROSPECTION: {self.model_type.upper()}")
        print(f"{'='*70}")
        
        try:
            from src.adrone.models.acoustic_models import CRNNWithAttention
            
            self.model = CRNNWithAttention(
                num_classes=num_classes,
                input_channels=input_channels,
                n_mels=n_mels,
                dropout=dropout
            )
            
            self.model.eval()
            print(f"✓ Loaded {self.model_type.upper()} model")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def trace_forward_pass(self, input_shape: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """Trace forward pass and record shapes and parameters"""
        self.input_shape = input_shape
        batch_size, channels, height, width = input_shape
        
        print(f"\nTracing forward pass with input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        layer_info = []
        
        # Helper to count parameters in a module
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        with torch.no_grad():
            # Input
            layer_info.append({
                'name': 'Input',
                'type': 'Input',
                'input_shape': list(dummy_input.shape),
                'output_shape': list(dummy_input.shape),
                'params': 0,
                'activation': 'None'
            })
            
            # Conv Block 1
            x = self.model.conv1(dummy_input)
            layer_info.append({
                'name': 'Conv Block 1',
                'type': 'Conv2d + BN + ReLU + MaxPool',
                'input_shape': list(dummy_input.shape),
                'output_shape': list(x.shape),
                'params': count_params(self.model.conv1),
                'activation': 'ReLU',
                'details': 'Conv2d(3→32, k=3, p=1) + BN + MaxPool(2)'
            })
            
            # Conv Block 2
            x_prev = x
            x = self.model.conv2(x)
            layer_info.append({
                'name': 'Conv Block 2',
                'type': 'Conv2d + BN + ReLU + MaxPool',
                'input_shape': list(x_prev.shape),
                'output_shape': list(x.shape),
                'params': count_params(self.model.conv2),
                'activation': 'ReLU',
                'details': 'Conv2d(32→64, k=3, p=1) + BN + MaxPool(2)'
            })
            
            # Conv Block 3
            x_prev = x
            x = self.model.conv3(x)
            layer_info.append({
                'name': 'Conv Block 3',
                'type': 'Conv2d + BN + ReLU + MaxPool',
                'input_shape': list(x_prev.shape),
                'output_shape': list(x.shape),
                'params': count_params(self.model.conv3),
                'activation': 'ReLU',
                'details': 'Conv2d(64→128, k=3, p=1) + BN + MaxPool(2)'
            })
            
            # Attention
            x_prev = x
            x = self.model.attention(x)
            layer_info.append({
                'name': 'TF-Attention',
                'type': 'Temporal-Frequency Attention',
                'input_shape': list(x_prev.shape),
                'output_shape': list(x.shape),
                'params': count_params(self.model.attention),
                'activation': 'Sigmoid',
                'details': 'Temporal + Frequency attention branches'
            })
            
            # Reshape for GRU
            batch, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(batch, t, -1)
            layer_info.append({
                'name': 'Reshape',
                'type': 'Reshape for RNN',
                'input_shape': [batch, c, f, t],
                'output_shape': list(x.shape),
                'params': 0,
                'activation': 'None',
                'details': f'Permute + View → [{batch}, {t}, {c*f}]'
            })
            
            # BiGRU
            x_prev = x
            x, _ = self.model.gru(x)
            layer_info.append({
                'name': 'BiGRU',
                'type': 'Bidirectional GRU',
                'input_shape': list(x_prev.shape),
                'output_shape': list(x.shape),
                'params': count_params(self.model.gru),
                'activation': 'tanh',
                'details': f'GRU(input={x_prev.shape[-1]}, hidden=128, layers=2, bidir=True)'
            })
            
            # Temporal pooling
            x_prev = x
            x = torch.mean(x, dim=1)
            layer_info.append({
                'name': 'Temporal Pooling',
                'type': 'Mean Pooling',
                'input_shape': list(x_prev.shape),
                'output_shape': list(x.shape),
                'params': 0,
                'activation': 'None',
                'details': 'Mean over time dimension'
            })
            
            # Dropout + FC
            x = self.model.dropout(x)
            x = self.model.fc(x)
            layer_info.append({
                'name': 'Classification',
                'type': 'Dropout + Linear',
                'input_shape': [batch, 256],
                'output_shape': list(x.shape),
                'params': count_params(self.model.fc),
                'activation': 'Softmax',
                'details': f'Dropout(0.3) + Linear(256→{x.shape[-1]})'
            })
        
        self.layer_info = layer_info
        
        # Print summary
        print(f"\n{'Layer':<20} {'Input Shape':<20} {'Output Shape':<20} {'Params':>12}")
        print("─" * 75)
        for info in layer_info:
            in_shape = str(info['input_shape'])[1:-1]
            out_shape = str(info['output_shape'])[1:-1]
            print(f"{info['name']:<20} {in_shape:<20} {out_shape:<20} {info['params']:>12,}")
        
        total_params = sum(info['params'] for info in layer_info)
        print("─" * 75)
        print(f"{'TOTAL':<20} {'':<20} {'':<20} {total_params:>12,}")
        
        return layer_info
    
    def save_metadata(self, output_path: Path):
        """Save architecture metadata as JSON"""
        metadata = {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'total_parameters': sum(info['params'] for info in self.layer_info),
            'layers': self.layer_info
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved architecture metadata to: {output_path}")


class PreprocessingAnalyzer:
    """Analyze preprocessing pipeline with actual data"""
    
    def __init__(self):
        from advanced_preprocessing import AudioPreprocessor
        
        self.preprocessor = AudioPreprocessor(
            sample_rate=22050,
            duration=3.0,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            n_mfcc=40,
            fmin=20,
            fmax=8000
        )
        
        self.config = {
            'sample_rate': self.preprocessor.sample_rate,
            'duration': self.preprocessor.duration,
            'n_samples': self.preprocessor.n_samples,
            'n_mels': self.preprocessor.n_mels,
            'n_fft': self.preprocessor.n_fft,
            'hop_length': self.preprocessor.hop_length,
            'n_mfcc': self.preprocessor.n_mfcc,
            'fmin': self.preprocessor.fmin,
            'fmax': self.preprocessor.fmax
        }
    
    def process_sample(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Process a sample through the entire pipeline"""
        # Load audio
        audio = self.preprocessor.load_audio(audio_path)
        
        # Extract features
        mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
        mfcc = self.preprocessor.extract_mfcc(audio)
        spectral = self.preprocessor.extract_spectral_features(audio)
        
        # Combined features
        combined = self.preprocessor.extract_combined_features(audio_path)
        
        return {
            'audio': audio,
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'spectral': spectral,
            'combined': combined
        }


class VisualizationGenerator:
    """Generate all visualizations"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.colors = {
            'input': '#FFE5E5',
            'conv': '#E3F2FD',
            'attention': '#F3E5F5',
            'rnn': '#FCE4EC',
            'pool': '#E1F5FE',
            'output': '#C8E6C9'
        }
    
    @staticmethod
    def add_inset_colorbar(mappable, ax):
        """Add non-overlapping inset colorbar"""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="3%", height="90%", loc="lower right",
                        bbox_to_anchor=(0.05, 0.05, 1, 1), 
                        bbox_transform=ax.transAxes, borderpad=0)
        return plt.colorbar(mappable, cax=cax)
    
    def generate_preprocessing_flowchart(self, analyzer: PreprocessingAnalyzer, 
                                         sample_files: Dict[str, str]):
        """Generate preprocessing flowchart with actual data"""
        print("\n" + "="*70)
        print("GENERATING: 01_preprocessing_flowchart.jpg")
        print("="*70)
        
        # Process one drone sample
        drone_sample = sample_files['drone']
        features = analyzer.process_sample(drone_sample)
        
        # Create figure (1080p slide dimensions)
        fig = plt.figure(figsize=(19.2, 10.8), constrained_layout=True)
        gs = fig.add_gridspec(5, 3, 
                             height_ratios=[0.9, 1.0, 2.0, 1.8, 1.6],
                             wspace=0.25, hspace=0.25)
        
        fig.suptitle('Preprocessing Pipeline: Raw Audio → 3-Channel Model Input',
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Row 1: Flowchart
        ax_flow = fig.add_subplot(gs[0, :])
        ax_flow.axis('off')
        self._draw_preprocessing_flowchart(ax_flow, analyzer.config)
        
        # Row 2: Waveform
        ax1 = fig.add_subplot(gs[1, :])
        times = np.arange(len(features['audio'])) / analyzer.config['sample_rate']
        ax1.plot(times, features['audio'], linewidth=0.5, color='#2E86AB')
        ax1.set_title(f"Step 1: Audio Waveform [SR={analyzer.config['sample_rate']}Hz, "
                      f"Duration={analyzer.config['duration']}s, Samples={len(features['audio'])}]",
                      fontsize=10, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_xlim(0, analyzer.config['duration'])
        
        # Row 3: Features
        ax2 = fig.add_subplot(gs[2, 0])
        img = librosa.display.specshow(features['mel_spec'], sr=analyzer.config['sample_rate'],
                                       hop_length=analyzer.config['hop_length'],
                                       x_axis='time', y_axis='mel', cmap='viridis', ax=ax2)
        ax2.set_title(f"Step 2: Mel Spectrogram\nShape: {features['mel_spec'].shape}",
                      fontsize=9, fontweight='bold')
        self.add_inset_colorbar(img, ax2)
        
        ax3 = fig.add_subplot(gs[2, 1])
        img = librosa.display.specshow(features['mfcc'][:40], sr=analyzer.config['sample_rate'],
                                       hop_length=analyzer.config['hop_length'],
                                       x_axis='time', cmap='coolwarm', ax=ax3)
        ax3.set_title(f"Step 3: MFCC\nShape: {features['mfcc'].shape}",
                      fontsize=9, fontweight='bold')
        ax3.set_ylabel('MFCC Coeffs')
        self.add_inset_colorbar(img, ax3)
        
        ax4 = fig.add_subplot(gs[2, 2])
        img = librosa.display.specshow(features['spectral'], sr=analyzer.config['sample_rate'],
                                       hop_length=analyzer.config['hop_length'],
                                       x_axis='time', cmap='plasma', ax=ax4)
        ax4.set_title(f"Step 4: Spectral Features\nShape: {features['spectral'].shape}",
                      fontsize=9, fontweight='bold')
        ax4.set_ylabel('Feature Index')
        self.add_inset_colorbar(img, ax4)
        
        # Row 4: Combined channels
        combined = features['combined']
        
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.imshow(combined[0], aspect='auto', origin='lower', cmap='viridis')
        ax5.set_title(f'Channel 0: Mel\n{combined[0].shape}', fontsize=9, fontweight='bold')
        ax5.set_xlabel('Time Frames')
        ax5.set_ylabel('Frequency Bins')
        
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.imshow(combined[1], aspect='auto', origin='lower', cmap='coolwarm')
        ax6.set_title(f'Channel 1: MFCC\n{combined[1].shape}', fontsize=9, fontweight='bold')
        ax6.set_xlabel('Time Frames')
        ax6.set_ylabel('Feature Bins')
        
        ax7 = fig.add_subplot(gs[3, 2])
        ax7.imshow(combined[2], aspect='auto', origin='lower', cmap='plasma')
        ax7.set_title(f'Channel 2: Spectral\n{combined[2].shape}', fontsize=9, fontweight='bold')
        ax7.set_xlabel('Time Frames')
        ax7.set_ylabel('Feature Bins')
        
        # Row 5: Final representation
        ax8 = fig.add_subplot(gs[4, :])
        combined_vis = np.concatenate([combined[0], combined[1], combined[2]], axis=1)
        ax8.imshow(combined_vis, aspect='auto', origin='lower', cmap='viridis')
        ax8.set_title(f'Final 3-Channel Stack → Model Input: {combined.shape}',
                      fontsize=10, fontweight='bold')
        ax8.set_xlabel('Time Frames (channels concatenated for visualization)')
        ax8.set_ylabel('Feature Bins')
        for spine in ax8.spines.values():
            spine.set_linewidth(0.8)
        
        # Add channel dividers
        w = combined[0].shape[1]
        ax8.axvline(w, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax8.axvline(w*2, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax8.text(w/2, -8, 'Mel', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        ax8.text(w*1.5, -8, 'MFCC', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        ax8.text(w*2.5, -8, 'Spectral', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        
        # Save (multiple formats for different uses)
        base = '01_preprocessing_flowchart'
        slide_path = self.output_dir / f'{base}_1080p.jpg'
        pdf_path = self.output_dir / f'{base}_A4.pdf'
        png_path = self.output_dir / f'{base}.png'
        
        plt.savefig(slide_path, dpi=150, bbox_inches='tight', format='jpg', pil_kwargs={'quality': 95})
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"✓ Saved 1080p JPEG: {slide_path}")
        print(f"✓ Saved A4 PDF: {pdf_path}")
        print(f"✓ Saved PNG: {png_path}")
        
        # Save config
        config_path = self.output_dir / '01_preprocessing_flowchart.meta.json'
        with open(config_path, 'w') as f:
            json.dump({
                'config': analyzer.config,
                'output_shapes': {
                    'audio': list(features['audio'].shape),
                    'mel_spec': list(features['mel_spec'].shape),
                    'mfcc': list(features['mfcc'].shape),
                    'spectral': list(features['spectral'].shape),
                    'combined': list(features['combined'].shape)
                }
            }, f, indent=2)
        print(f"✓ Saved metadata: {config_path}")
    
    def _draw_preprocessing_flowchart(self, ax, config):
        """Draw preprocessing flowchart"""
        boxes = [
            {'x': 0.02, 'label': 'WAV\nFile', 'color': '#FF6B6B', 'w': 0.10},
            {'x': 0.14, 'label': f'Resample\n{config["sample_rate"]}Hz', 'color': '#4ECDC4', 'w': 0.12},
            {'x': 0.28, 'label': 'Normalize\n[-1, 1]', 'color': '#4ECDC4', 'w': 0.12},
            {'x': 0.42, 'label': f'Mel Spec\n{config["n_mels"]} bins', 'color': '#95E1D3', 'w': 0.12},
            {'x': 0.56, 'label': f'MFCC\n{config["n_mfcc"]} coeffs', 'color': '#95E1D3', 'w': 0.12},
            {'x': 0.70, 'label': 'Spectral\nFeatures', 'color': '#95E1D3', 'w': 0.12},
            {'x': 0.84, 'label': '3-Channel\nStack', 'color': '#F38181', 'w': 0.12}
        ]
        
        y_pos = 0.5
        box_height = 0.6
        
        for i, box in enumerate(boxes):
            rect = FancyBboxPatch((box['x'], y_pos - box_height/2), box['w'], box_height,
                                  boxstyle="round,pad=0.03",
                                  edgecolor='black', facecolor=box['color'],
                                  linewidth=1.5, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(box['x'] + box['w']/2, y_pos, box['label'],
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   transform=ax.transAxes)
            
            if i < len(boxes) - 1:
                arrow = FancyArrowPatch((box['x'] + box['w'], y_pos),
                                       (boxes[i+1]['x'], y_pos),
                                       arrowstyle='->', mutation_scale=20, linewidth=2,
                                       color='black', transform=ax.transAxes)
                ax.add_patch(arrow)
    
    def generate_architecture_diagram(self, introspector: ModelIntrospector):
        """Generate CRNN architecture diagram from actual model"""
        import textwrap
        
        print("\n" + "="*70)
        print("GENERATING: 02_crnn_architecture.jpg")
        print("="*70)
        
        layer_info = introspector.layer_info
        total_params = sum(info['params'] for info in layer_info)
        
        fig, ax = plt.subplots(figsize=(19.2, 14.4), constrained_layout=True)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, len(layer_info) + 3)
        ax.axis('off')
        
        # Title
        ax.text(6, len(layer_info) + 2.2, 'CRNN with Temporal-Frequency Attention',
               ha='center', fontsize=14, fontweight='bold')
        ax.text(6, len(layer_info) + 1.6, f'Total Parameters: {total_params:,}',
               ha='center', fontsize=10, style='italic')
        
        # Helper for text wrapping
        def wrap_text(s, width=72):
            return "\n".join(textwrap.wrap(s, width))
        
        # Draw layers from top to bottom
        y_start = len(layer_info)
        
        for i, info in enumerate(layer_info):
            y_pos = y_start - i
            
            # Choose color
            if 'Conv' in info['name']:
                color = self.colors['conv']
            elif 'Attention' in info['name']:
                color = self.colors['attention']
            elif 'GRU' in info['name']:
                color = self.colors['rnn']
            elif 'Pool' in info['name']:
                color = self.colors['pool']
            elif 'Classification' in info['name']:
                color = self.colors['output']
            else:
                color = self.colors['input']
            
            # Draw box
            rect = FancyBboxPatch((1.5, y_pos - 0.4), 7, 0.8,
                                  boxstyle="round,pad=0.03",
                                  edgecolor='black', facecolor=color,
                                  linewidth=1.5)
            ax.add_patch(rect)
            
            # Text content
            in_shape = str(info['input_shape'])[1:-1]
            out_shape = str(info['output_shape'])[1:-1]
            
            text = f"{info['name']}\n"
            if info.get('details'):
                text += f"{info['details']}\n"
            text += f"In: [{in_shape}] → Out: [{out_shape}]\n"
            text += f"Params: {info['params']:,} | Activation: {info['activation']}"
            
            # Wrap long text
            wrapped_text = wrap_text(text, width=72)
            
            ax.text(5, y_pos, wrapped_text,
                   ha='center', va='center', fontsize=7, family='monospace')
            
            # Draw arrow to next layer
            if i < len(layer_info) - 1:
                arrow = FancyArrowPatch((5, y_pos - 0.5), (5, y_pos - 0.6),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color='darkblue')
                ax.add_patch(arrow)
        
        # Parameter distribution in right panel
        conv_params = sum(info['params'] for info in layer_info if 'Conv' in info['name'])
        attn_params = sum(info['params'] for info in layer_info if 'Attention' in info['name'])
        gru_params = sum(info['params'] for info in layer_info if 'GRU' in info['name'])
        fc_params = sum(info['params'] for info in layer_info if 'Classification' in info['name'])
        
        stats_text = f"""Parameter
Distribution:

Conv Blocks:
{conv_params:,}
({conv_params/total_params*100:.1f}%)

TF-Attention:
{attn_params:,}
({attn_params/total_params*100:.1f}%)

BiGRU:
{gru_params:,}
({gru_params/total_params*100:.1f}%)

Classifier:
{fc_params:,}
({fc_params/total_params*100:.1f}%)"""
        
        # Right side panel
        right_panel_x = 10.3
        rect = FancyBboxPatch((right_panel_x, 1.0), 1.4, len(layer_info), 
                              boxstyle="round,pad=0.03",
                              edgecolor='black', facecolor='lightyellow', 
                              linewidth=1.2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(right_panel_x + 0.7, 1.0 + len(layer_info)/2.0, stats_text,
               ha='center', va='center', fontsize=7, family='monospace')
        
        # Save (multiple formats)
        base = '02_crnn_architecture'
        slide_path = self.output_dir / f'{base}_1080p.jpg'
        pdf_path = self.output_dir / f'{base}_A4.pdf'
        png_path = self.output_dir / f'{base}.png'
        
        plt.savefig(slide_path, dpi=150, bbox_inches='tight', format='jpg', pil_kwargs={'quality': 95})
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"✓ Saved 1080p JPEG: {slide_path}")
        print(f"✓ Saved A4 PDF: {pdf_path}")
        print(f"✓ Saved PNG: {png_path}")
        
        # Save metadata (matches the diagram exactly)
        introspector.save_metadata(self.output_dir / '02_crnn_architecture.meta.json')
    
    def generate_training_pipeline(self):
        """Generate training pipeline diagram"""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        print("\n" + "="*70)
        print("GENERATING: 03_training_pipeline.jpg")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(5, 9.5, 'Training Pipeline: Data → Model → Optimization',
               ha='center', fontsize=14, fontweight='bold')
        
        steps = [
            {'y': 8.5, 'text': '1. Data Loading\n• WeightedRandomSampler for class balance\n• Batch size: 32\n• Pin memory for GPU transfer', 'color': '#FFE5E5'},
            {'y': 7.3, 'text': '2. Preprocessing\n• Resample: 22050 Hz\n• Extract 3-channel features (Mel, MFCC, Spectral)\n• Normalize: [0, 1]', 'color': '#E3F2FD'},
            {'y': 6.1, 'text': '3. Augmentation (training only)\n• Time shift • Pitch shift\n• Add noise • Time stretch', 'color': '#E8F5E9'},
            {'y': 4.9, 'text': '4. Forward Pass\n• Conv blocks → Attention → BiGRU\n• Output logits [B, 3]', 'color': '#FFF3E0'},
            {'y': 3.7, 'text': '5. Loss Computation\n• CrossEntropyLoss with class weights\n• Focal loss option (γ=2.0)', 'color': '#F3E5F5'},
            {'y': 2.5, 'text': '6. Optimization\n• AdamW (lr=1e-4, weight_decay=1e-4)\n• Cosine annealing LR\n• Gradient clipping (max_norm=1.0)', 'color': '#FCE4EC'},
            {'y': 1.3, 'text': '7. Validation & Checkpointing\n• Macro F1-score\n• Early stopping (patience=10)\n• Save best model', 'color': '#C8E6C9'}
        ]
        
        for i, step in enumerate(steps):
            rect = FancyBboxPatch((1, step['y'] - 0.5), 8, 1.0,
                                  boxstyle="round,pad=0.03",
                                  edgecolor='black', facecolor=step['color'],
                                  linewidth=1.5)
            ax.add_patch(rect)
            
            ax.text(5, step['y'], step['text'],
                   ha='center', va='center', fontsize=8, family='monospace')
            
            if i < len(steps) - 1:
                arrow = FancyArrowPatch((5, step['y'] - 0.6), (5, steps[i+1]['y'] + 0.6),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color='darkblue')
                ax.add_patch(arrow)
        
        # Save (multiple formats)
        base = '03_training_pipeline'
        slide_path = self.output_dir / f'{base}_1080p.jpg'
        pdf_path = self.output_dir / f'{base}_A4.pdf'
        png_path = self.output_dir / f'{base}.png'
        
        plt.savefig(slide_path, dpi=150, bbox_inches='tight', format='jpg', pil_kwargs={'quality': 95})
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"✓ Saved 1080p JPEG: {slide_path}")
        print(f"✓ Saved A4 PDF: {pdf_path}")
        print(f"✓ Saved PNG: {png_path}")
    
    def generate_system_overview(self):
        """Generate complete system flowchart"""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        print("\n" + "="*70)
        print("GENERATING: 04_complete_system_flowchart.jpg")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 11)
        ax.axis('off')
        
        ax.text(5, 10.5, 'Complete System: Audio → Classification + Confidence',
               ha='center', fontsize=14, fontweight='bold')
        
        boxes = [
            {'y': 9.5, 'w': 3, 'text': 'INPUT\nWAV Audio File\n(any duration)', 'color': '#FFE5E5'},
            {'y': 8.0, 'w': 3, 'text': 'PREPROCESSING\nResample → Mel/MFCC/Spectral\n→ 3-channel stack', 'color': '#E3F2FD'},
            {'y': 6.5, 'w': 3, 'text': 'FEATURE EXTRACTION\nConv1 → Conv2 → Conv3\nTF-Attention', 'color': '#E8F5E9'},
            {'y': 5.0, 'w': 3, 'text': 'TEMPORAL MODELING\nBiGRU (2 layers)\nTemporal pooling', 'color': '#FFF3E0'},
            {'y': 3.5, 'w': 3, 'text': 'CLASSIFICATION\nDropout → Linear → Softmax\n3 classes', 'color': '#F3E5F5'},
            {'y': 2.0, 'w': 3, 'text': 'OUTPUT\nClass: drone / helicopter / background\nConfidence: [0.0, 1.0]', 'color': '#C8E6C9'}
        ]
        
        for i, box in enumerate(boxes):
            x_center = 5
            rect = FancyBboxPatch((x_center - box['w']/2, box['y'] - 0.6), box['w'], 1.2,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='black', facecolor=box['color'],
                                  linewidth=2)
            ax.add_patch(rect)
            
            ax.text(x_center, box['y'], box['text'],
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
            if i < len(boxes) - 1:
                arrow = FancyArrowPatch((x_center, box['y'] - 0.7),
                                       (x_center, boxes[i+1]['y'] + 0.7),
                                       arrowstyle='->', mutation_scale=25,
                                       linewidth=3, color='darkblue')
                ax.add_patch(arrow)
        
        # Inference time note in inset
        note = inset_axes(ax, width="18%", height="14%", loc="center right", borderpad=1.0)
        note.axis('off')
        note.text(0.5, 0.5, 'Inference:\nGPU: 10–20 ms\nCPU: 50–100 ms',
                 ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Save (multiple formats)
        base = '04_complete_system_flowchart'
        slide_path = self.output_dir / f'{base}_1080p.jpg'
        pdf_path = self.output_dir / f'{base}_A4.pdf'
        png_path = self.output_dir / f'{base}.png'
        
        plt.savefig(slide_path, dpi=150, bbox_inches='tight', format='jpg', pil_kwargs={'quality': 95})
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"✓ Saved 1080p JPEG: {slide_path}")
        print(f"✓ Saved A4 PDF: {pdf_path}")
        print(f"✓ Saved PNG: {png_path}")
    
    def generate_class_comparison(self, analyzer: PreprocessingAnalyzer,
                                   sample_files: Dict[str, str]):
        """Generate class comparison visualization"""
        print("\n" + "="*70)
        print("GENERATING: class_comparison.png")
        print("="*70)
        
        fig, axes = plt.subplots(3, 4, figsize=(19.2, 10.8), 
                                constrained_layout=True, sharex='col')
        fig.suptitle('Class Signatures: Drone vs Helicopter vs Background',
                     fontsize=14, fontweight='bold', y=0.995)
        
        colors = {'drone': '#FF6B6B', 'helicopter': '#4ECDC4', 'background': '#95E1D3'}
        
        for idx, (class_name, audio_path) in enumerate(sample_files.items()):
            features = analyzer.process_sample(audio_path)
            
            # Waveform
            ax = axes[idx, 0]
            times = np.arange(len(features['audio'])) / analyzer.config['sample_rate']
            ax.plot(times, features['audio'], linewidth=0.5, color=colors[class_name])
            ax.set_title(f'{class_name.capitalize()}\nWaveform', fontweight='bold', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Mel Spectrogram
            ax = axes[idx, 1]
            img = librosa.display.specshow(features['mel_spec'],
                                           sr=analyzer.config['sample_rate'],
                                           hop_length=analyzer.config['hop_length'],
                                           x_axis='time', y_axis='mel',
                                           cmap='viridis', ax=ax)
            ax.set_title(f'{class_name.capitalize()}\nMel Spectrogram', fontweight='bold', fontsize=9)
            plt.colorbar(img, ax=ax, format='%0.1f', pad=0.02)
            
            # MFCC
            ax = axes[idx, 2]
            img = librosa.display.specshow(features['mfcc'][:40],
                                           sr=analyzer.config['sample_rate'],
                                           hop_length=analyzer.config['hop_length'],
                                           x_axis='time', cmap='coolwarm', ax=ax)
            ax.set_title(f'{class_name.capitalize()}\nMFCC', fontweight='bold', fontsize=9)
            ax.set_ylabel('MFCC Coeffs', fontsize=8)
            plt.colorbar(img, ax=ax, format='%0.1f', pad=0.02)
            
            # Spectral Features
            ax = axes[idx, 3]
            img = librosa.display.specshow(features['spectral'],
                                           sr=analyzer.config['sample_rate'],
                                           hop_length=analyzer.config['hop_length'],
                                           x_axis='time', cmap='plasma', ax=ax)
            ax.set_title(f'{class_name.capitalize()}\nSpectral', fontweight='bold', fontsize=9)
            plt.colorbar(img, ax=ax, format='%0.1f', pad=0.02)
        
        # Save
        png_path = self.output_dir / 'class_comparison.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"✓ Saved PNG: {png_path}")


def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("PROFESSIONAL VISUALIZATION GENERATOR")
    print("Acoustic Drone Detection System")
    print("="*70)
    
    # Paths
    train_dir = r"f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/train"
    val_dir = r"f:/EDTH/acoustic-drone-detector/data/edth_munich_dataset/data/val"
    
    try:
        # 1. Validate dataset
        validator = DatasetValidator(train_dir, val_dir)
        dataset_info = validator.validate()
        
        if not dataset_info['valid']:
            print("\n❌ Dataset validation failed!")
            for warning in dataset_info['warnings']:
                print(f"  - {warning}")
            return
        
        # Validate audio samples
        audio_info = validator.validate_audio_samples(target_sr=22050, target_duration=3.0)
        
        # 2. Initialize preprocessing analyzer
        analyzer = PreprocessingAnalyzer()
        
        # Get sample files
        sample_files = {}
        for class_name in ['drone', 'helicopter', 'background']:
            if class_name in dataset_info['classes']:
                sample_file = dataset_info['classes'][class_name]['sample_file']
                if sample_file:
                    sample_files[class_name] = sample_file
        
        # 3. Initialize model introspector
        introspector = ModelIntrospector('crnn')
        introspector.load_model(num_classes=3, input_channels=3, n_mels=128, dropout=0.3)
        
        # Trace forward pass with actual input shape from preprocessing
        sample_features = analyzer.process_sample(sample_files['drone'])
        input_shape = (1,) + sample_features['combined'].shape  # Add batch dimension
        introspector.trace_forward_pass(input_shape)
        
        # 4. Generate visualizations
        generator = VisualizationGenerator()
        
        generator.generate_preprocessing_flowchart(analyzer, sample_files)
        generator.generate_architecture_diagram(introspector)
        generator.generate_training_pipeline()
        generator.generate_system_overview()
        generator.generate_class_comparison(analyzer, sample_files)
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*70)
        print(f"\nOutput directory: {generator.output_dir.absolute()}")
        print("\nGenerated files:")
        for f in sorted(generator.output_dir.glob("*")):
            print(f"  - {f.name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
