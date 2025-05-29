from flask import Flask, render_template, request, flash, send_file, redirect, url_for
import os
import numpy as np
import cv2
import tifffile
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import math

app = Flask(__name__)
app.secret_key = 'frap_secret_key'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analytical_model(t, k, K0, D, rho_e):
    TI = sum(((-K0)**j) / (math.factorial(j) * np.sqrt(1 + j)) for j in range(4))
    TD = [sum(((-K0)**j) / math.factorial(j) /
              np.sqrt((1 + j) * (1 + (8 * D / rho_e**2) * t_)) for j in range(4)) for t_ in t]
    return [k * td + (1 - k) * TI for td in TD]

def load_image_convert_png(filepath):
    img = tifffile.imread(filepath)
    if img.dtype == np.uint16:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    png_path = filepath.replace('.tif', '.png')
    cv2.imwrite(png_path, img)
    return img, os.path.basename(png_path)

def suggest_bleach_regions(img):
    strip = img[:, 65:75]
    row_means = np.mean(strip, axis=1)
    drop_idx = np.argmin(row_means[:100])
    summary = (
        f"Suggested columns (MATLAB default): 65-75\n"
        f"Pre-bleach rows (fixed for analysis): 0-49\n"
        f"Post-bleach rows (fixed for analysis): 50-{img.shape[0]-1}\n"
        f"Note: Bleach point detected at row {drop_idx} (display only)"
    )
    return 65, 75, 0, img.shape[0]-1, summary, drop_idx

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {
        'image_file': '', 'height': 0, 'width': 0,
        'suggestion_text': '', 'default_x1': 65, 'default_x2': 75,
        'default_y1': 0, 'default_y2': 0, 'tiff_file': '', 'bleach_start': 50
    }

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.lower().endswith('.tif'):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                img_array, png_filename = load_image_convert_png(filepath)
                x1, x2, y1, y2, suggestion, _ = suggest_bleach_regions(img_array)
                result.update({
                    'image_file': png_filename,
                    'height': img_array.shape[0],
                    'width': img_array.shape[1],
                    'suggestion_text': suggestion,
                    'default_x1': x1, 'default_x2': x2,
                    'default_y1': y1, 'default_y2': y2,
                    'tiff_file': filename
                })
            except Exception as e:
                flash(f"Error processing image: {e}")
    return render_template('single_page_cropper.html', **result)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        filename = request.form.get('file')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.png', '.tif'))
        if not os.path.exists(filepath):
            flash("TIFF file not found on server.")
            return redirect('/')

        x1 = int(request.form.get('x1') or 65)
        x2 = int(request.form.get('x2') or 75)

        img = tifffile.imread(filepath)
        height, width = img.shape
        def_lines = slice(x1, x2 + 1)
        pre_blch_lines = img[0:50, def_lines]
        avg_pr_blch = np.mean(pre_blch_lines)
        max_row = min(height, 512)
        time = np.arange(0, max_row - 50)
        blch_lines = img[50:max_row, def_lines]
        mean_blch_lines = np.mean(blch_lines, axis=1)
        norm_mean_blch_lines = mean_blch_lines / avg_pr_blch

        min_len = min(len(time), len(norm_mean_blch_lines))
        time = time[:min_len]
        norm_mean_blch_lines = norm_mean_blch_lines[:min_len]

        K0_fixed, rho_e_fixed = 0.5, 0.5
        def fit_func(t, k, D):
            return 1 - np.array(analytical_model(t, k, K0_fixed, D, rho_e_fixed))

        try:
            popt, _ = curve_fit(fit_func, time, norm_mean_blch_lines, p0=[0.8, 10], bounds=([0, 1], [1, 100]))
            fitted_k, fitted_D = popt
            fit_curve = fit_func(time, fitted_k, fitted_D)
        except Exception:
            fitted_k, fitted_D = 0.8, 10
            fit_curve = fit_func(time, fitted_k, fitted_D)

        default_curve = fit_func(time, 0.8, 10)

        fig_d = go.Figure()
        for D_val in np.linspace(1, 50, 10)[::2]:
            fig_d.add_trace(go.Scatter(x=time, y=fit_func(time, fitted_k, D_val), mode='lines', name=f'D={D_val:.1f}'))
        fig_d.add_trace(go.Scatter(x=time, y=norm_mean_blch_lines, mode='markers', name='Experimental Data', marker=dict(color='red')))

        fig_k = go.Figure()
        for k_val in np.linspace(0.1, 1.0, 10)[::2]:
            fig_k.add_trace(go.Scatter(x=time, y=fit_func(time, k_val, fitted_D), mode='lines', name=f'k={k_val:.1f}'))
        fig_k.add_trace(go.Scatter(x=time, y=norm_mean_blch_lines, mode='markers', name='Experimental Data', marker=dict(color='red')))

        fig_final = go.Figure()
        fig_final.add_trace(go.Scatter(x=time, y=norm_mean_blch_lines, mode='markers', name='Experimental Data', marker=dict(color='red')))
        fig_final.add_trace(go.Scatter(x=time, y=fit_curve, mode='lines', name='Fitted Curve', line=dict(color='blue')))
        fig_final.add_trace(go.Scatter(x=time, y=default_curve, mode='lines', name='Default Curve', line=dict(color='gray', dash='dash')))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"frap_data_{timestamp}.csv"
        plot_name = f"frap_plot_{timestamp}.png"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_name)
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_name)

        try:
            pd.DataFrame({
                'Time': time,
                'Normalized_Intensity': norm_mean_blch_lines,
                'Fitted_Curve': fit_curve,
                'Default_Curve': default_curve
            }).to_csv(csv_path, index=False)
            pio.write_image(fig_final, plot_path, width=800, height=500)
        except Exception as save_err:
            print(f"Saving failed: {save_err}")
            csv_name = ''
            plot_name = ''

        interpretation = f"""
        <h4>Analysis Results:</h4>
        <ul>
            <li><b>Mobile Fraction (k):</b> {fitted_k:.3f}</li>
            <li><b>Diffusion Coefficient (D):</b> {fitted_D:.3f}</li>
            <li><b>Pre-bleach avg intensity:</b> {avg_pr_blch:.2f}</li>
            <li><b>Columns analyzed:</b> {x1}-{x2}</li>
            <li><b>Frames:</b> {len(time)}</li>
        </ul>
        """

        return render_template(
            'results.html',
            chart_d=pio.to_html(fig_d, include_plotlyjs=False, full_html=False),
            chart_k=pio.to_html(fig_k, include_plotlyjs=False, full_html=False),
            chart_final=pio.to_html(fig_final, include_plotlyjs=True, full_html=False),
            interpretation=interpretation,
            csv_file=csv_name,
            plot_file=plot_name
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        flash(f"Analysis error: {e}")
        return redirect('/')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

