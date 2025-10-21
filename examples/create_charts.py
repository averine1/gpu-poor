"""
Generate visual charts for gpu-poor results
Run after: python -m examples.demo <model>
Creates charts in results/charts/
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load all results from results/*.json"""
    results_dir = Path("results")
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        if json_file.name == "all_results.json":
            continue
        
        model_name = json_file.stem
        with open(json_file) as f:
            results[model_name] = json.load(f)
    
    return results

def create_compression_chart(results, output_path="results/charts/compression.png"):
    """Memory compression comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    original = [results[m]['original_size'] for m in models]
    compressed = [results[m]['compressed_size'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, compressed, width, label='Quantized', color='#2ecc71', alpha=0.8)
    
    # Add reduction percentages on top
    for i, (orig, comp) in enumerate(zip(original, compressed)):
        reduction = (1 - comp/orig) * 100
        ax.text(i, max(orig, comp) + 50, f'-{reduction:.0f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('gpu-poor: Memory Compression Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Created: {output_path}")
    plt.close()

def create_speed_chart(results, output_path="results/charts/speed.png"):
    """Speed comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    speedups = [results[m]['speedup_x'] for m in models]
    colors = ['#2ecc71' if s >= 0.95 else '#f39c12' if s >= 0.8 else '#e74c3c' for s in speedups]
    
    bars = ax.bar(models, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0×)')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{speedup:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('gpu-poor: Speed Performance by Model Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Created: {output_path}")
    plt.close()

def create_quality_chart(results, output_path="results/charts/quality.png"):
    """Quality metrics (Perplexity + BLEU)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(results.keys())
    
    # Perplexity degradation
    ppl_changes = [results[m].get('perplexity_degradation_pct', 0) for m in models]
    colors_ppl = ['#2ecc71' if abs(p) < 5 else '#f39c12' if abs(p) < 10 else '#e74c3c' for p in ppl_changes]
    
    bars1 = ax1.bar(models, ppl_changes, color=colors_ppl, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.axhline(y=5, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='5% threshold')
    ax1.axhline(y=-5, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    for bar, change in zip(bars1, ppl_changes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3 if height > 0 else height - 0.3,
                f'{change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Perplexity Change (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Perplexity Degradation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticklabels(models, rotation=15, ha='right')
    
    # BLEU scores
    bleu_scores = [results[m].get('bleu_score', 0) * 100 for m in models]  # Convert to percentage
    colors_bleu = ['#2ecc71' if b > 95 else '#f39c12' if b > 90 else '#e74c3c' for b in bleu_scores]
    
    bars2 = ax2.bar(models, bleu_scores, color=colors_bleu, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=95, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Excellent (>95)')
    ax2.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Good (>90)')
    
    for bar, bleu in zip(bars2, bleu_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{bleu:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('BLEU Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Text Generation Similarity', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticklabels(models, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Created: {output_path}")
    plt.close()

def create_summary_dashboard(results, output_path="results/charts/summary.png"):
    """Combined dashboard with all metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    models = list(results.keys())
    
    # 1. Compression
    ax1 = fig.add_subplot(gs[0, 0])
    compression_ratios = [(1 - results[m]['compressed_size']/results[m]['original_size']) * 100 for m in models]
    bars1 = ax1.bar(models, compression_ratios, color='#3498db', alpha=0.8, edgecolor='black')
    for bar, ratio in zip(bars1, compression_ratios):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{ratio:.0f}%', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('Compression (%)', fontweight='bold')
    ax1.set_title('Memory Reduction', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    
    # 2. Speed
    ax2 = fig.add_subplot(gs[0, 1])
    speedups = [results[m]['speedup_x'] for m in models]
    colors = ['#2ecc71' if s >= 0.95 else '#f39c12' if s >= 0.8 else '#e74c3c' for s in speedups]
    bars2 = ax2.bar(models, speedups, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    for bar, speedup in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{speedup:.2f}×', ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontweight='bold')
    ax2.set_title('Speed Performance', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    
    # 3. BLEU
    ax3 = fig.add_subplot(gs[1, 0])
    bleu_scores = [results[m].get('bleu_score', 0) * 100 for m in models]
    bars3 = ax3.bar(models, bleu_scores, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax3.axhline(y=95, color='green', linestyle=':', alpha=0.5)
    for bar, bleu in zip(bars3, bleu_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bleu:.1f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylabel('BLEU Score (%)', fontweight='bold')
    ax3.set_title('Generation Quality', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_xticklabels(models, rotation=15, ha='right')
    
    # 4. Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Model', 'Comp.', 'Speed', 'BLEU', 'Status'])
    for m in models:
        comp = f"{(1 - results[m]['compressed_size']/results[m]['original_size']) * 100:.0f}%"
        speed = f"{results[m]['speedup_x']:.2f}×"
        bleu = f"{results[m].get('bleu_score', 0):.3f}"
        status = '✅' if results[m].get('bleu_score', 0) > 0.95 else '⚠️'
        table_data.append([m, comp, speed, bleu, status])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary', fontsize=13, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('gpu-poor: Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Created: {output_path}")
    plt.close()

def main():
    """Generate all charts"""
    print("\n" + "="*60)
    print("Creating Visual Charts for gpu-poor")
    print("="*60)
    
    # Create charts directory
    charts_dir = Path("results/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No results found. Run demo first:")
        print("   python -m examples.demo gpt2")
        return
    
    print(f"\n📊 Loaded results for {len(results)} models")
    print(f"   Models: {', '.join(results.keys())}")
    
    # Generate charts
    print("\n🎨 Generating charts...")
    create_compression_chart(results)
    create_speed_chart(results)
    create_quality_chart(results)
    create_summary_dashboard(results)
    
    print("\n" + "="*60)
    print("✅ All charts created in results/charts/")
    print("="*60)
    print("\nFiles created:")
    print("  - compression.png  (memory comparison)")
    print("  - speed.png        (speed performance)")
    print("  - quality.png      (perplexity + BLEU)")
    print("  - summary.png      (combined dashboard)")
    print("\nUse these in your README, social media, and presentations!")

if __name__ == "__main__":
    main()