"""Chạy notebook thống kê và cập nhật output, không cần khởi động Jupyter kernel."""
import base64
import contextlib
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat


def main():
    path = Path(__file__).with_name('data_static.ipynb')
    notebook = nbformat.read(path, as_version=4)
    namespace = {'__name__': '__main__'}
    original_savefig = plt.savefig

    def savefig_with_retry(*args, **kwargs):
        for attempt in range(4):
            try:
                return original_savefig(*args, **kwargs)
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(1)

    plt.savefig = savefig_with_retry
    counter = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        counter += 1
        cell.execution_count = counter
        outputs = []
        stream = io.StringIO()

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format='png', bbox_inches='tight')
                outputs.append(nbformat.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                    'text/plain': repr(figure),
                }))
            plt.close('all')

        original_show = plt.show
        plt.show = show
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell.source, f'{path}:cell{counter}', 'exec'), namespace)
        finally:
            plt.show = original_show
        if stream.getvalue():
            outputs.insert(0, nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue()))
        cell.outputs = outputs
    nbformat.write(notebook, path)
    print(f'Đã chạy {counter} ô và cập nhật {path}')


if __name__ == '__main__':
    main()
