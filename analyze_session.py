import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_COLORS = {
    'HIGH STRESS': '#FF0000',
    'EXCITEMENT': '#FFA500',
    'RELAXED': '#00FF00',
    'ALERT': '#FF8C00',
    'FATIGUE': '#0064FF',
    'UNKNOWN': '#808080',
}

EMOTION_MARKERS = {
    'Happy': ('o', '#00FF00'),
    'Neutral': ('s', '#808080'),
    'Sad': ('v', '#0000FF'),
    'Angry': ('^', '#FF0000'),
    'Fear': ('D', '#800080'),
    'Surprise': ('*', '#FFD700'),
}


def load_session(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed = {}
            for key, value in row.items():
                if value == '':
                    processed[key] = None
                elif key in ['elapsed', 'bpm', 'sqi', 'rmssd', 'sdnn',
                             'pnn50', 'lf_hf', 'breathing_rate',
                             'emotion_confidence', 'hrv_stress_score', 'timestamp']:
                    try:
                        processed[key] = float(value)
                    except ValueError:
                        processed[key] = None
                else:
                    processed[key] = value
            data.append(processed)
    return data


def compute_statistics(data):
    stats = {
        'duration': 0,
        'total_rows': len(data),
        'bpm': {'values': [], 'min': None, 'max': None, 'mean': None, 'std': None},
        'sqi': {'values': [], 'mean': None},
        'hrv_stress_score': {'values': [], 'mean': None},
        'rmssd': {'values': [], 'mean': None},
        'sdnn': {'values': [], 'mean': None},
        'state_time': {},
        'emotion_counts': {},
    }

    if not data:
        return stats

    first_elapsed = data[0].get('elapsed', 0) or 0
    last_elapsed = data[-1].get('elapsed', 0) or 0
    stats['duration'] = last_elapsed - first_elapsed

    for row in data:
        for metric in ['bpm', 'sqi', 'hrv_stress_score', 'rmssd', 'sdnn']:
            val = row.get(metric)
            if val is not None and val > 0:
                stats[metric]['values'].append(val)

        state = row.get('state', 'UNKNOWN')
        if state:
            stats['state_time'][state] = stats['state_time'].get(state, 0) + 1

        emotion = row.get('emotion', 'Neutral')
        if emotion:
            stats['emotion_counts'][emotion] = stats['emotion_counts'].get(emotion, 0) + 1

    for metric in ['bpm', 'sqi', 'hrv_stress_score', 'rmssd', 'sdnn']:
        values = stats[metric]['values']
        if values:
            stats[metric]['mean'] = np.mean(values)
            stats[metric]['std'] = np.std(values)
            stats[metric]['min'] = np.min(values)
            stats[metric]['max'] = np.max(values)

    total_state_rows = sum(stats['state_time'].values())
    if total_state_rows > 0:
        for state in stats['state_time']:
            count = stats['state_time'][state]
            stats['state_time'][state] = {
                'count': count,
                'percent': (count / total_state_rows) * 100,
                'seconds': count / 30
            }

    return stats


def print_statistics(stats, filepath):
    print("\n" + "=" * 60)
    print(f"SESSION: {os.path.basename(filepath)}")
    print("=" * 60)

    duration = stats['duration']
    mins = int(duration // 60)
    secs = int(duration % 60)
    print(f"\nDuration: {mins}m {secs}s ({stats['total_rows']} samples)")

    bpm = stats['bpm']
    if bpm['mean']:
        print(f"\nHeart Rate: {bpm['mean']:.1f} BPM (range: {bpm['min']:.0f}-{bpm['max']:.0f})")

    sqi = stats['sqi']
    if sqi['mean']:
        print(f"Signal Quality: {sqi['mean']:.2f}")

    rmssd = stats['rmssd']
    hrv_stress = stats['hrv_stress_score']
    if rmssd['mean']:
        print(f"HRV RMSSD: {rmssd['mean']:.1f} ms")
    if hrv_stress['mean']:
        print(f"Stress Score: {hrv_stress['mean']:.2f}")

    print(f"\nState Distribution:")
    for state, info in sorted(stats['state_time'].items(),
                               key=lambda x: x[1]['percent'], reverse=True):
        print(f"  {state}: {info['percent']:.1f}%")

    print("=" * 60)


def create_timeline_figure(data, stats, filepath):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Session: {os.path.basename(filepath)}', fontsize=14, fontweight='bold')

    elapsed = [row.get('elapsed', 0) or 0 for row in data]
    bpm = [row.get('bpm') for row in data]
    sqi = [row.get('sqi') for row in data]
    hrv_stress = [row.get('hrv_stress_score') for row in data]
    states = [row.get('state', 'UNKNOWN') for row in data]
    emotions = [row.get('emotion', 'Neutral') for row in data]

    ax1 = axes[0]
    ax1.set_ylabel('BPM')
    ax1.set_title('Heart Rate')

    prev_state = None
    band_start = 0
    for i, (t, state) in enumerate(zip(elapsed, states)):
        if state != prev_state and prev_state is not None:
            color = STATE_COLORS.get(prev_state, '#808080')
            ax1.axvspan(band_start, t, alpha=0.3, color=color, linewidth=0)
            band_start = t
        elif prev_state is None:
            band_start = t
        prev_state = state
    if prev_state and elapsed:
        color = STATE_COLORS.get(prev_state, '#808080')
        ax1.axvspan(band_start, elapsed[-1], alpha=0.3, color=color, linewidth=0)

    valid_bpm = [(t, b) for t, b in zip(elapsed, bpm) if b is not None]
    if valid_bpm:
        t_bpm, v_bpm = zip(*valid_bpm)
        ax1.plot(t_bpm, v_bpm, 'k-', linewidth=1)
        ax1.set_ylim(min(v_bpm) - 10, max(v_bpm) + 10)

    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.set_ylabel('SQI')
    ax2.set_title('Signal Quality')

    valid_sqi = [(t, s) for t, s in zip(elapsed, sqi) if s is not None]
    if valid_sqi:
        t_sqi, v_sqi = zip(*valid_sqi)
        ax2.fill_between(t_sqi, 0, v_sqi, alpha=0.4, color='green')
        ax2.plot(t_sqi, v_sqi, 'g-', linewidth=1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.set_ylabel('Stress')
    ax3.set_title('HRV Stress Score')

    valid_hrv = [(t, h) for t, h in zip(elapsed, hrv_stress) if h is not None]
    if valid_hrv:
        t_hrv, v_hrv = zip(*valid_hrv)
        colors = ['green' if v < 0.4 else 'orange' if v < 0.7 else 'red' for v in v_hrv]
        ax3.scatter(t_hrv, v_hrv, c=colors, s=10, alpha=0.6)
        ax3.plot(t_hrv, v_hrv, 'k-', linewidth=0.5, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.set_ylabel('Emotion')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Emotions')

    emotion_order = ['Happy', 'Surprise', 'Neutral', 'Sad', 'Fear', 'Angry']
    emotion_to_y = {e: i for i, e in enumerate(emotion_order)}

    for emotion in emotion_order:
        points = [(t, emotion_to_y[emotion]) for t, e in zip(elapsed, emotions)
                  if e == emotion]
        if points:
            t_pts, y_pts = zip(*points)
            marker, color = EMOTION_MARKERS.get(emotion, ('o', '#808080'))
            ax4.scatter(t_pts, y_pts, marker=marker, c=color, s=20, alpha=0.6)

    ax4.set_yticks(range(len(emotion_order)))
    ax4.set_yticklabels(emotion_order)
    ax4.set_ylim(-0.5, len(emotion_order) - 0.5)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def create_summary_figure(stats, filepath):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'Summary: {os.path.basename(filepath)}', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    if stats['state_time']:
        labels = list(stats['state_time'].keys())
        sizes = [stats['state_time'][s]['percent'] for s in labels]
        colors = [STATE_COLORS.get(s, '#808080') for s in labels]
        ax1.pie(sizes, labels=labels, colors=colors,
                autopct=lambda p: f'{p:.1f}%' if p > 5 else '', startangle=90)
        ax1.set_title('States')

    ax2 = axes[1]
    if stats['emotion_counts']:
        emotions = list(stats['emotion_counts'].keys())
        counts = list(stats['emotion_counts'].values())
        total = sum(counts)
        percentages = [(c / total) * 100 for c in counts]

        sorted_data = sorted(zip(emotions, percentages), key=lambda x: x[1], reverse=True)
        emotions, percentages = zip(*sorted_data)

        colors = [EMOTION_MARKERS.get(e, ('o', '#808080'))[1] for e in emotions]
        ax2.barh(emotions, percentages, color=colors, alpha=0.7)
        ax2.set_xlabel('%')
        ax2.set_title('Emotions')

    ax3 = axes[2]
    ax3.axis('off')

    lines = []
    duration = stats['duration']
    lines.append(f"Duration: {int(duration//60)}m {int(duration%60)}s")

    bpm = stats['bpm']
    if bpm['mean']:
        lines.append(f"BPM: {bpm['mean']:.1f}")

    sqi = stats['sqi']
    if sqi['mean']:
        lines.append(f"SQI: {sqi['mean']:.2f}")

    rmssd = stats['rmssd']
    if rmssd['mean']:
        lines.append(f"RMSSD: {rmssd['mean']:.1f} ms")

    ax3.text(0.1, 0.9, '\n'.join(lines), transform=ax3.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax3.set_title('Metrics')

    plt.tight_layout()
    return fig


def analyze_session(filepath, output_dir=None, show=True):
    print(f"\nLoading {filepath}...")
    data = load_session(filepath)

    if not data:
        print(f"No data in {filepath}")
        return None

    stats = compute_statistics(data)
    print_statistics(stats, filepath)

    fig_timeline = create_timeline_figure(data, stats, filepath)
    fig_summary = create_summary_figure(stats, filepath)

    if output_dir is None:
        output_dir = os.path.dirname(filepath)

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    timeline_path = os.path.join(output_dir, f"{base_name}_timeline.png")
    summary_path = os.path.join(output_dir, f"{base_name}_summary.png")

    fig_timeline.savefig(timeline_path, dpi=150, bbox_inches='tight')
    fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')

    print(f"\nSaved: {timeline_path}")
    print(f"Saved: {summary_path}")

    if show:
        plt.show()
    else:
        plt.close('all')

    return stats


def find_latest_session(logs_dir="logs"):
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return None

    csv_files = list(logs_path.glob("session_*.csv"))
    if not csv_files:
        return None

    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(csv_files[0])


def main():
    parser = argparse.ArgumentParser(description='Analyze session logs')
    parser.add_argument('files', nargs='*', help='CSV files')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('-o', '--output-dir', type=str)

    args = parser.parse_args()

    files_to_analyze = []

    if args.all:
        logs_path = Path("logs")
        if logs_path.exists():
            files_to_analyze = sorted(logs_path.glob("session_*.csv"))
    elif args.files:
        files_to_analyze = [Path(f) for f in args.files]
    else:
        latest = find_latest_session()
        if latest:
            files_to_analyze = [Path(latest)]
        else:
            print("No session files found")
            sys.exit(1)

    if not files_to_analyze:
        print("No files to analyze")
        sys.exit(1)

    for filepath in files_to_analyze:
        if not filepath.exists():
            print(f"Not found: {filepath}")
            continue

        analyze_session(str(filepath), output_dir=args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
