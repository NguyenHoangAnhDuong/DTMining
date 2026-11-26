import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finish(fig, path):
    """Lưu hình thay vì show để pipeline chạy được trên máy không có GUI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
