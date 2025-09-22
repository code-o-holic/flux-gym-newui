import os
import sys
import json
import yaml
import toml
import shutil
import platform
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Callable, Optional

from nicegui import ui, app
from PIL import Image
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download
import signal
from nicegui import app as nicegui_app


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')
DATASETS_DIR = os.path.join(ROOT, 'datasets')

# Global process handle for stop endpoint
CURRENT_PROC = None

# discover sd-scripts utils (as in app.py)
sys.path.insert(0, ROOT)
sd_scripts_path = os.path.join(ROOT, 'sd-scripts')
if sd_scripts_path not in sys.path:
    sys.path.append(sd_scripts_path)

_HAS_SD_SCRIPTS = False
_SD_IMPORT_ERROR = ''
try:
    import train_network  # type: ignore
    from library import flux_train_utils  # type: ignore
    _HAS_SD_SCRIPTS = hasattr(train_network, 'setup_parser')
except Exception as e:
    _SD_IMPORT_ERROR = str(e)
    # Fallback: try import by absolute file paths
    try:
        import importlib.util
        tn_path = os.path.join(sd_scripts_path, 'train_network.py')
        if os.path.isfile(tn_path):
            spec = importlib.util.spec_from_file_location('train_network', tn_path)
            if spec and spec.loader:
                train_network = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(train_network)  # type: ignore
        lib_path = os.path.join(sd_scripts_path, 'library')
        if lib_path not in sys.path:
            sys.path.append(lib_path)
        from library import flux_train_utils  # type: ignore
        _HAS_SD_SCRIPTS = hasattr(train_network, 'setup_parser')
    except Exception as e2:
        _SD_IMPORT_ERROR = f"{_SD_IMPORT_ERROR or ''} | {e2}"

# Global styles to improve readability and avoid overflow in Advanced panel
ui.add_head_html(
    """
<style>
.advanced-scope .q-field__label,
.advanced-scope .q-field__messages,
.advanced-scope .q-checkbox__label {
  white-space: normal !important;
  word-break: break-word !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.advanced-scope .q-field { width: 100%; }
</style>
"""
)

ui.add_head_html(
    """
<style>
/* Full-height layout */
html, body { height: 100%; margin: 0; padding: 0; }
.nicegui-content { padding: 0 !important; }
#app { height: 100%; }
.fullheight { min-height: 100vh; }
/* Top pane cards fill their row (60vh) and scroll internally */
.pane-card { height: 100%; display: flex; flex-direction: column; margin: 0; }
.pane-scroll { flex: 1 1 auto; overflow: auto; }
/* Terminal row spacing */
.terminal-pane { width: 100%; margin-top: 12px; }
/* Splitter visual cue (if any remain) */
.q-splitter__separator { background: rgba(0,0,0,0.05); }
.q-splitter__separator:before { content: ''; display:block; width: 8px; height: 100%; background: rgba(79,155,227,0.6); border-radius: 2px; margin: 0 auto; }
</style>
"""
)


def list_runs() -> List[str]:
    try:
        items = [os.path.join(OUTPUTS_DIR, name) for name in os.listdir(OUTPUTS_DIR)]
        runs = [p for p in items if os.path.isdir(p) and os.path.basename(p) != 'sample']
        runs.sort(key=lambda p: os.path.getctime(p), reverse=True)
        return runs
    except Exception:
        return []


def read_models() -> Dict[str, Any]:
    models_path = os.path.join(ROOT, 'models.yaml')
    try:
        with open(models_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_slugify(text: str) -> str:
    try:
        from slugify import slugify as _slugify
        return _slugify(text)
    except Exception:
        # Minimal fallback for Python 3
        try:
            norm = unicodedata.normalize('NFKD', text)
        except Exception:
            norm = str(text)
        ascii_text = norm.encode('ascii', 'ignore').decode('ascii')
        ascii_text = re.sub(r'[^a-zA-Z0-9\-_.]+', '-', ascii_text)
        ascii_text = ascii_text.strip('-_.')
        ascii_text = re.sub(r'-{2,}', '-', ascii_text)
        return ascii_text or 'run'


def open_gradio_ui() -> None:
    try:
        ui.run_javascript('window.open("http://localhost:7860", "_blank")')
    except Exception:
        pass


def compute_expected_steps(max_epochs: int, repeats: int, images: int) -> int:
    try:
        return int(max_epochs) * int(repeats) * int(images)
    except Exception:
        return 0


def save_resized_image(src_path: str, dst_path: str, size: int) -> None:
    with Image.open(src_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size / width) * height)
        else:
            new_height = size
            new_width = int((size / height) * width)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(dst_path)


def generate_dataset(folder: str, uploaded_files: List[str], resize_to: int, caption_rows: List[Dict[str, Any]] = None) -> None:
    ensure_dir(folder)
    captions_map: Dict[str, str] = {}
    if caption_rows:
        for row in caption_rows:
            try:
                img_base = os.path.splitext(os.path.basename(row['image']))[0]
                captions_map[img_base] = (row['caption'].value or '').strip()
            except Exception:
                continue
    for src in uploaded_files:
        try:
            filename = os.path.basename(src)
            dst = os.path.join(folder, filename)
            shutil.copy(src, dst)
            ext = os.path.splitext(dst)[-1].lower()
            if ext != '.txt':
                save_resized_image(dst, dst, resize_to)
                # write caption file
                base = os.path.splitext(os.path.basename(dst))[0]
                caption_txt = captions_map.get(base, '')
                if caption_txt:
                    cap_path = os.path.join(folder, f"{base}.txt")
                    try:
                        with open(cap_path, 'w', encoding='utf-8') as cf:
                            cf.write(caption_txt)
                    except Exception:
                        pass
        except Exception:
            continue


def build_train_script(
    base_model_key: str,
    models_cfg: Dict[str, Any],
    output_name: str,
    seed: int,
    workers: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: float,
    max_train_epochs: int,
    save_every_n_epochs: int,
    timestep_sampling: str,
    guidance_scale: float,
    vram: str,
    sample_prompts_present: bool,
    sample_every_n_steps: int,
    advanced_flags: List[str],
) -> str:
    line_break = "^" if platform.system().lower().startswith('win') else "\\"

    model_cfg = models_cfg.get(base_model_key, {})
    repo = model_cfg.get('repo')
    file_name = model_cfg.get('file')
    if base_model_key in ('flux-dev', 'flux-schnell'):
        model_folder = 'models/unet'
    else:
        model_folder = f"models/unet/{repo}"
    pretrained_model_path = os.path.join(model_folder, file_name)

    clip_path = 'models/clip/clip_l.safetensors'
    t5_path = 'models/clip/t5xxl_fp16.safetensors'
    ae_path = 'models/vae/ae.sft'

    output_dir_path = os.path.join(OUTPUTS_DIR, output_name)
    output_dir_q = f'"{output_dir_path}"'
    dataset_toml_q = f'"{os.path.join(output_dir_path, "dataset.toml")}"'
    sample_prompts_q = f'"{os.path.join(output_dir_path, "sample_prompts.txt")}"'
    logs_dir_q = f'"{os.path.join(output_dir_path, "logs")}"'

    if vram == '16G':
        optimizer = f"--optimizer_type adafactor {line_break}\n  --optimizer_args \"relative_step=False\" \"scale_parameter=False\" \"warmup_init=False\" {line_break}\n  --lr_scheduler constant_with_warmup {line_break}\n  --max_grad_norm 0.0 {line_break}"
    elif vram == '12G':
        optimizer = f"--optimizer_type adafactor {line_break}\n  --optimizer_args \"relative_step=False\" \"scale_parameter=False\" \"warmup_init=False\" {line_break}\n  --split_mode {line_break}\n  --network_args \"train_blocks=single\" {line_break}\n  --lr_scheduler constant_with_warmup {line_break}\n  --max_grad_norm 0.0 {line_break}"
    else:
        # Use regular AdamW instead of 8bit version for better compatibility
        optimizer = f"--optimizer_type AdamW {line_break}"

    sample = ""
    if sample_prompts_present and sample_every_n_steps > 0:
        sample = f"--sample_prompts={sample_prompts_q} --sample_every_n_steps=\"{sample_every_n_steps}\" {line_break}\n  "

    base = f"""accelerate launch {line_break}
  --config_file accelerate_config.yaml {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path \"{pretrained_model_path}\" {line_break}
  --clip_l \"{clip_path}\" {line_break}
  --t5xxl \"{t5_path}\" {line_break}
  --ae \"{ae_path}\" {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  --network_alpha {network_alpha} {line_break}
  {optimizer}
  {sample}--learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {dataset_toml_q} {line_break}
  --output_dir {output_dir_q} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}
  --logging_dir {logs_dir_q} {line_break}
  --log_with tensorboard"""

    if advanced_flags:
        base += "\n  " + f" {line_break}\n  ".join(advanced_flags)
    return base


def write_and_launch_run(
    lora_name: str,
    concept_sentence: str,
    models_cfg: Dict[str, Any],
    base_model: str,
    vram: str,
    repeats_per_image: int,
    max_epochs: int,
    resize_to: int,
    batch_size: int,
    sample_prompts: str,
    sample_every_n_steps: int,
    seed: int,
    workers: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: float,
    guidance_scale: float,
    timestep_sampling: str,
    advanced_flags: List[str],
    uploaded_files: List[str],
    caption_rows: List[Dict[str, Any]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    output_name = safe_slugify(lora_name)
    output_dir = os.path.join(OUTPUTS_DIR, output_name)
    ensure_dir(output_dir)

    # dataset
    dataset_folder = os.path.join(ROOT, 'datasets', output_name)
    ensure_dir(dataset_folder)
    # Include user-edited captions
    generate_dataset(dataset_folder, uploaded_files, resize_to, caption_rows=caption_rows)

    # Ensure required models are present (download if missing)
    try:
        models_yaml = read_models()
        model_cfg = models_yaml.get(base_model, {})
        repo = model_cfg.get('repo')
        file_name = model_cfg.get('file')
        # unet
        if base_model in ('flux-dev', 'flux-schnell'):
            unet_folder = os.path.join(ROOT, 'models', 'unet')
        else:
            unet_folder = os.path.join(ROOT, 'models', 'unet', repo or '')
        os.makedirs(unet_folder, exist_ok=True)
        unet_path = os.path.join(unet_folder, file_name or '')
        if file_name and not os.path.exists(unet_path):
            if log_fn: log_fn(f'Downloading base model: {base_model} ...')
            hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=file_name)
            if log_fn: log_fn('Base model download complete')
        # vae
        vae_folder = os.path.join(ROOT, 'models', 'vae')
        os.makedirs(vae_folder, exist_ok=True)
        vae_path = os.path.join(vae_folder, 'ae.sft')
        if not os.path.exists(vae_path):
            if log_fn: log_fn('Downloading VAE ...')
            hf_hub_download(repo_id='cocktailpeanut/xulf-dev', local_dir=vae_folder, filename='ae.sft')
            if log_fn: log_fn('VAE download complete')
        # clip/text encoders
        clip_folder = os.path.join(ROOT, 'models', 'clip')
        os.makedirs(clip_folder, exist_ok=True)
        if not os.path.exists(os.path.join(clip_folder, 'clip_l.safetensors')):
            if log_fn: log_fn('Downloading CLIP L ...')
            hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', local_dir=clip_folder, filename='clip_l.safetensors')
            if log_fn: log_fn('CLIP L download complete')
        if not os.path.exists(os.path.join(clip_folder, 't5xxl_fp16.safetensors')):
            if log_fn: log_fn('Downloading T5 XXL fp16 ...')
            hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', local_dir=clip_folder, filename='t5xxl_fp16.safetensors')
            if log_fn: log_fn('T5 XXL download complete')
    except Exception:
        pass

    # toml
    toml_path = os.path.join(output_dir, 'dataset.toml')
    config = {
        'general': {
            'shuffle_caption': False,
            'caption_extension': '.txt',
            'keep_tokens': 1,
        },
        'datasets': [{
            'resolution': resize_to,
            'batch_size': int(batch_size),
            'keep_tokens': 1,
            'subsets': [{
                'image_dir': dataset_folder,
                'class_tokens': concept_sentence,
                'num_repeats': repeats_per_image,
            }],
        }],
    }
    with open(toml_path, 'w', encoding='utf-8') as f:
        toml.dump(config, f)

    # sample prompts
    sample_prompts_path = os.path.join(output_dir, 'sample_prompts.txt')
    with open(sample_prompts_path, 'w', encoding='utf-8') as f:
        f.write(sample_prompts or '')

    # build script
    script = build_train_script(
        base_model, models_cfg, output_name, seed, workers, learning_rate, network_dim, network_alpha,
        max_epochs, 4, timestep_sampling, guidance_scale, vram,
        len(sample_prompts.strip()) > 0, sample_every_n_steps, advanced_flags,
    )

    # save script
    is_windows = platform.system().lower().startswith('win')
    file_type = 'bat' if is_windows else 'sh'
    script_path = os.path.join(output_dir, f'train.{file_type}')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

    # run.json for comparisons
    run_json = {
        'lora_name': lora_name,
        'base_model': base_model,
        'vram': vram,
        'repeats_per_image': repeats_per_image,
        'max_epochs': max_epochs,
        'resize_to': resize_to,
        'batch_size': int(batch_size),
        'seed': seed,
        'workers': workers,
        'learning_rate': learning_rate,
        'network_dim': network_dim,
        'network_alpha': network_alpha,
        'guidance_scale': guidance_scale,
        'timestep_sampling': timestep_sampling,
        'advanced_flags': advanced_flags,
    }
    with open(os.path.join(output_dir, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(run_json, f, indent=2)

    return script_path, toml_path
def _sanitize_help(text: str) -> str:
    try:
        if not text:
            return ''
        # Prefer content before a bilingual divider
        if ' / ' in text:
            text = text.split(' / ', 1)[0]
        # Keep ASCII only to remove JP/CN characters
        text = ''.join(ch for ch in text if ord(ch) < 128)
        # Collapse whitespace
        return ' '.join(text.split())
    except Exception:
        return text



def build_advanced_form() -> Tuple[Dict[str, Any], Dict[str, str]]:
    inputs: Dict[str, Any] = {}
    help_texts: Dict[str, str] = {}
    if not _HAS_SD_SCRIPTS:
        msg = 'sd-scripts parser not found; advanced options unavailable.'
        if _SD_IMPORT_ERROR:
            msg += f" (Import error: {_SD_IMPORT_ERROR})"
        ui.label(msg).classes('text-amber-600')
        return inputs, help_texts

    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)

    # Map argparse destinations to nice groups
    categories: Dict[str, str] = {
        'optimizer': 'Optimization', 'max_grad_norm': 'Optimization', 'optimizer_args': 'Optimization',
        'lr_scheduler': 'Scheduler', 'lr_warmup_steps': 'Scheduler',
        'mixed_precision': 'Precision/Memory', 'save_precision': 'Precision/Memory', 'gradient_checkpointing': 'Precision/Memory', 'highvram': 'Precision/Memory',
        'timestep_sampling': 'Sampling', 'guidance_scale': 'Sampling',
        'cache_latents_to_disk': 'Data', 'cache_text_encoder_outputs': 'Data', 'cache_text_encoder_outputs_to_disk': 'Data',
    }

    # Collect metadata first
    meta_groups: Dict[str, List[Dict[str, Any]]] = {}
    for action in parser._actions:
        dest = getattr(action, 'dest', None)
        if not dest or dest == 'help':
            continue
        if dest in {'pretrained_model_name_or_path','clip_l','t5xxl','ae','network_module','network_dim','learning_rate','output_dir','output_name'}:
            continue
        flag = action.option_strings[0] if action.option_strings else f'--{dest}'
        info = _sanitize_help(action.help or '')
        help_texts[dest] = info
        cat = categories.get(dest, 'Misc')
        if cat not in meta_groups:
            meta_groups[cat] = []
        meta_groups[cat].append({
            'dest': dest,
            'flag': flag,
            'help': info,
            'is_bool': str(action.type) == 'None',
        })

    # Render grouped sections and create components inside
    for cat, items in sorted(meta_groups.items(), key=lambda x: x[0]):
        with ui.expansion(cat).classes('w-full advanced-scope'):
            for item in items:
                if item['is_bool']:
                    comp = ui.checkbox(item['flag']).classes('w-full')
                    if item['help']:
                        comp.tooltip = item['help']
                else:
                    comp = ui.input(label=item['flag']).props('filled').classes('w-full')
                    if item['help']:
                        help_text = item['help'].replace('"', "'")
                        comp.props(f'hint="{help_text}" persistent-hint')
                comp.classes('q-mb-md')
                inputs[item['dest']] = comp
    return inputs, help_texts


def top_navbar() -> None:
    with ui.header().style('background: linear-gradient(90deg, #4f9be3, #6dc4a3); color: white'):
        with ui.row().classes('items-center justify-between w-full'):
            ui.label('FluxGym: New UI').classes('text-2xl font-bold')
            with ui.row():
                ui.button('Open Gradio UI', on_click=open_gradio_ui, color='white').props('flat outline')


def page_train() -> None:
    models_cfg = read_models()
    model_names = list(models_cfg.keys())

    state: Dict[str, Any] = {}

    # Install client-side beforeunload warning and stop beacon
    ui.run_javascript(
        """
        if (!window.__fg_beforeunload_installed__) {
            window.__fg_beforeunload_installed__ = true;
            window.isDirty = false;
            window.trainingRunning = false;
            window.addEventListener('beforeunload', function(e) {
                if (window.trainingRunning || window.isDirty) {
                    e.preventDefault();
                    e.returnValue = 'Training is running or you have unsaved changes. Leaving will stop training and reset.';
                    return e.returnValue;
                }
            });
            window.addEventListener('unload', function() {
                if (window.trainingRunning) {
                    try { navigator.sendBeacon('/stop_training'); } catch (e) {}
                }
            });
        }
        """
    )

    def mark_dirty() -> None:
        ui.run_javascript('window.isDirty = true')

    def set_training_flag(flag: bool) -> None:
        try:
            ui.run_javascript(f'window.trainingRunning = {str(flag).lower()}')
        except Exception:
            # If we're outside UI context, just ignore - not critical
            pass

    # Main vertical layout: top row (3 panes) + bottom terminal
    with ui.column().classes('w-full fullheight').style('max-width: 100vw; margin: 0'):
        # Top: three panes
        with ui.row().classes('w-full').style('gap: 16px; height: 60vh; align-items: stretch; overflow: hidden'):
            # Step 1
            with ui.card().classes('pane-card p-4').style('flex: 1 1 0; min-width: 0; height: 60vh; overflow: auto'):
                ui.markdown('### Step 1. LoRA Setup')
                with ui.column().classes('pane-scroll'):
                    state['lora_name'] = ui.input(label='The name of your LoRA', placeholder='e.g.: Persian Miniature Painting style, Cat Toy')
                    state['lora_name'].on('change', lambda e: mark_dirty())
                    state['concept_sentence'] = ui.input(label='Trigger word/sentence', placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'")
                    state['concept_sentence'].on('change', lambda e: mark_dirty())
                    state['base_model'] = ui.select(options=model_names, value=(model_names[0] if model_names else None), label='Base model (from models.yaml)')
                    state['base_model'].on('change', lambda e: mark_dirty())
                    ui.label('VRAM')
                    state['vram'] = ui.radio(['20G', '16G', '12G'], value='20G')
                    state['vram'].on('change', lambda e: mark_dirty())
                    with ui.row():
                        state['repeats'] = ui.number(label='Repeat trains per image', value=10)
                        state['repeats'].on('change', lambda e: mark_dirty())
                        state['max_epochs'] = ui.number(label='Max Train Epochs', value=16)
                        state['max_epochs'].on('change', lambda e: mark_dirty())
                    state['expected_steps'] = ui.number(label='Expected training steps', value=0)
                    state['expected_steps'].disable()
                    state['sample_prompts'] = ui.textarea(label='Sample Image Prompts (Separate with new lines)', placeholder='').props('autogrow')
                    state['sample_prompts'].on('change', lambda e: mark_dirty())
                    with ui.row():
                        state['sample_every_n_steps'] = ui.number(label='Sample Image Every N Steps', value=0)
                        state['sample_every_n_steps'].on('change', lambda e: mark_dirty())
                        state['resize_to'] = ui.number(label='Resize dataset images', value=512)
                        state['resize_to'].on('change', lambda e: mark_dirty())
                    ui.separator()
                    ui.markdown('#### Main Training Params').classes('text-primary')
                    with ui.row():
                        state['network_dim'] = ui.number(label='network_dim', value=4)
                        state['network_dim'].on('change', lambda e: mark_dirty())
                        state['network_alpha'] = ui.number(label='network_alpha', value=1.0)
                        state['network_alpha'].on('change', lambda e: mark_dirty())
                        state['batch_size'] = ui.number(label='batch_size', value=1)
                        state['batch_size'].on('change', lambda e: mark_dirty())
                    state['learning_rate'] = ui.input(label='learning_rate', value='8e-4')
                    state['learning_rate'].on('change', lambda e: mark_dirty())
                    with ui.expansion('Advanced parameters (from sd-scripts)').classes('q-mt-md'):
                        build_advanced_form()

            # Step 2
            with ui.card().classes('pane-card p-4').style('flex: 1 1 0; min-width: 0; height: 60vh; overflow: auto'):
                ui.markdown('### Step 2. Dataset')
                with ui.column().classes('pane-scroll'):
                    state['uploaded_paths']: List[str] = []
                    state['caption_rows']: List[Dict[str, Any]] = []
                    grid = ui.grid(columns=2).classes('gap-3 w-full')

                    def load_captions() -> None:
                        grid.clear(); state['caption_rows'].clear()
                        trigger = state['concept_sentence'].value or ''
                        files = list(state['uploaded_paths'])
                        images = [f for f in files if os.path.splitext(f)[1].lower() != '.txt']
                        txts = {os.path.splitext(os.path.basename(f))[0]: f for f in files if os.path.splitext(f)[1].lower() == '.txt'}
                        for img_path in images:
                            base = os.path.splitext(os.path.basename(img_path))[0]
                            cap_text = ''
                            if base in txts:
                                try:
                                    with open(txts[base], 'r', encoding='utf-8') as tf:
                                        cap_text = tf.read().strip()
                                except Exception:
                                    cap_text = ''
                            if not cap_text and trigger:
                                cap_text = trigger
                            with grid:
                                ui.image(img_path).classes('w-40 h-40 object-cover')
                                cap_input = ui.textarea(value=cap_text).props('autogrow').classes('w-full')
                                state['caption_rows'].append({'image': img_path, 'caption': cap_input})

                    STAGE_DIR = os.path.join(OUTPUTS_DIR, '_uploads'); ensure_dir(STAGE_DIR)

                    def handle_upload(e) -> None:
                        try:
                            name = getattr(e, 'name', None) or 'file'
                            content = getattr(e, 'content', None)
                            if content is None: return
                            dst = os.path.join(STAGE_DIR, name)
                            data = content.read(); open(dst, 'wb').write(data)
                            try: content.seek(0)
                            except Exception: pass
                            state['uploaded_paths'].append(dst)
                            render_file_list(); load_captions(); mark_dirty()
                        except Exception:
                            ui.notify('Upload failed. Try again.', type='warning')

                    state['uploads'] = ui.upload(label='Upload images (+ optional .txt captions)', multiple=True, on_upload=handle_upload)
                    try: state['uploads'].props('auto-upload')
                    except Exception: pass

                    file_list = ui.column().classes('w-full q-mt-md')

                    def remove_file(path: str) -> None:
                        try:
                            if os.path.exists(path): os.remove(path)
                        except Exception: pass
                        base, ext = os.path.splitext(os.path.basename(path))
                        counterpart = os.path.join(STAGE_DIR, (base + ('.png' if ext.lower()=='.txt' else '.txt')))
                        try:
                            if os.path.exists(counterpart): os.remove(counterpart)
                        except Exception: pass
                        state['uploaded_paths'] = [p for p in state['uploaded_paths'] if os.path.abspath(p) not in {os.path.abspath(path), os.path.abspath(counterpart)}]
                        state['caption_rows'] = [r for r in state['caption_rows'] if os.path.abspath(r['image']) not in {os.path.abspath(path), os.path.abspath(counterpart)}]
                        render_file_list()
                        load_captions()

                    def render_file_list() -> None:
                        file_list.clear(); imgs=[p for p in state['uploaded_paths'] if os.path.splitext(p)[1].lower()!='.txt']; txts=[p for p in state['uploaded_paths'] if os.path.splitext(p)[1].lower()=='.txt']
                        for p in imgs+txts:
                            with file_list:
                                with ui.row().classes('items-center justify-between w-full'):
                                    ui.label(os.path.basename(p)).classes('ellipsis')
                                    ui.button('Remove', on_click=lambda _p=p: remove_file(_p)).props('outline color=negative size=sm')

                    state['last_trigger'] = ''

                    def apply_trigger_to_empty() -> None:
                        trigger = state['concept_sentence'].value or ''
                        for row in state['caption_rows']:
                            if not (row['caption'].value or '').strip(): row['caption'].value = trigger

                    def apply_trigger_to_all() -> None:
                        trigger = state['concept_sentence'].value or ''
                        for row in state['caption_rows']: row['caption'].value = trigger

                    def on_trigger_change(e=None) -> None:
                        new_tr = state['concept_sentence'].value or ''
                        old_tr = state.get('last_trigger', '')
                        for row in state['caption_rows']:
                            cur = row['caption'].value or ''
                            if old_tr and cur.startswith(old_tr+' '): row['caption'].value = (new_tr + ' ' + cur[len(old_tr)+1:]).strip()
                            elif not cur.strip(): row['caption'].value = new_tr
                        state['last_trigger'] = new_tr

                    state['concept_sentence'].on('change', on_trigger_change)
                    with ui.row().classes('gap-2 q-mt-sm'):
                        ui.button('Refresh captions list', on_click=load_captions).props('outline')
                        ui.button('Fill empty with trigger', on_click=apply_trigger_to_empty).props('outline')
                        ui.button('Fill ALL with trigger', on_click=apply_trigger_to_all).props('outline color=negative')

            # Step 3
            with ui.card().classes('pane-card p-4').style('flex: 1 1 0; min-width: 0; height: 60vh; overflow: auto'):
                ui.markdown('### Step 3. Train Config')
                state['proc_handle'] = None
                # Start/Stop at top
                state['train_btn'] = ui.button('Start training', color='positive').classes('q-mb-sm')
                with ui.tabs().classes('w-full') as t3:
                    tab_script = ui.tab('Script')
                    tab_config = ui.tab('Config')
                with ui.tab_panels(t3, value=tab_script).classes('w-full pane-scroll'):
                    with ui.tab_panel(tab_script):
                        state['script_code'] = ui.code('', language='bash').classes('w-full').style('height: 46vh')
                    with ui.tab_panel(tab_config):
                        state['config_code'] = ui.code('', language='toml').classes('w-full').style('height: 46vh')

                # small helper to safely update code widgets
                def set_code_content(widget, text: str) -> None:
                    try:
                        widget.set_content(text)
                        return
                    except Exception:
                        pass
                    try:
                        widget.set_text(text)
                        return
                    except Exception:
                        pass
                    try:
                        widget.content = text
                    except Exception:
                        pass

                def ui_set_script(text: str) -> None:
                    set_code_content(state['script_code'], text)

                def ui_set_config(text: str) -> None:
                    set_code_content(state['config_code'], text)

                def ui_log(line: str) -> None:
                    try:
                        state['terminal_log'].push(line)
                    except Exception:
                        pass

                # Button behavior
                def set_btn_running(running: bool) -> None:
                    if running:
                        state['train_btn'].text = 'Stop training'
                        state['train_btn'].props('color=negative')
                    else:
                        state['train_btn'].text = 'Start training'
                        state['train_btn'].props('color=positive')

                def toggle_training() -> None:
                    if state.get('proc_handle') is not None:
                        try:
                            stop_process(state['proc_handle'])
                            ui_log("[STOP] Training stopped by user")
                        except Exception as e:
                            ui_log(f"[STOP ERROR] {e}")
                        state['proc_handle'] = None
                        global CURRENT_PROC
                        CURRENT_PROC = None
                        set_btn_running(False)
                        set_training_flag(False)
                        return
                    start_training_impl()

                def start_training_impl() -> None:
                    if not (state['lora_name'].value or '').strip():
                        ui.notify('LoRA name is required', type='negative'); return
                    if len(state.get('uploaded_paths', [])) == 0:
                        ui.notify('Please upload at least one image', type='negative'); return
                    uploads = list(state.get('uploaded_paths', []))
                    def log_fn(msg: str) -> None:
                        try: state['terminal_log'].push(msg)
                        except Exception: pass
                    # Preview
                    output_name_preview = safe_slugify(state['lora_name'].value or 'run')
                    try:
                        script_preview = build_train_script(
                            state['base_model'].value or (model_names[0] if model_names else ''),
                            models_cfg,
                            output_name_preview,
                            42,
                            2,
                            state['learning_rate'].value or '8e-4',
                            int(state['network_dim'].value or 4),
                            float(state['network_alpha'].value or 1.0),
                            int(state['max_epochs'].value or 16),
                            4,
                            'shift',
                            1.0,
                            state['vram'].value or '20G',
                            len((state['sample_prompts'].value or '').strip()) > 0,
                            int(state['sample_every_n_steps'].value or 0),
                            [],
                        )
                        cfg_preview = {
                            'general': {
                                'shuffle_caption': False,
                                'caption_extension': '.txt',
                                'keep_tokens': 1,
                            },
                            'datasets': [{
                                'resolution': int(state['resize_to'].value or 512),
                                'batch_size': int(state['batch_size'].value or 1),
                                'keep_tokens': 1,
                                'subsets': [{
                                    'image_dir': os.path.join(ROOT, 'datasets', output_name_preview),
                                    'class_tokens': state['concept_sentence'].value or '',
                                    'num_repeats': int(state['repeats'].value or 10),
                                }],
                            }],
                        }
                        ui_set_script(script_preview)
                        ui_set_config(toml.dumps(cfg_preview))
                    except Exception:
                        pass

                    # Capture the current UI context for the worker thread
                    from nicegui import context
                    try:
                        current_slot = context.slot
                    except Exception:
                        current_slot = None
                    
                    def worker():
                        # Set button state
                        if current_slot:
                            with current_slot:
                                set_btn_running(True)
                                set_training_flag(True)
                        else:
                            # Fallback without context
                            try:
                                state['train_btn'].text = 'Stop training'
                                state['train_btn'].props('color=negative')
                            except Exception:
                                pass
                        
                        try:
                            script_path, toml_path = write_and_launch_run(
                                lora_name=state['lora_name'].value or 'run',
                                concept_sentence=state['concept_sentence'].value or '',
                                models_cfg=models_cfg,
                                base_model=state['base_model'].value or (model_names[0] if model_names else ''),
                                vram=state['vram'].value or '20G',
                                repeats_per_image=int(state['repeats'].value or 10),
                                max_epochs=int(state['max_epochs'].value or 16),
                                resize_to=int(state['resize_to'].value or 512),
                                batch_size=int(state['batch_size'].value or 1),
                                sample_prompts=state['sample_prompts'].value or '',
                                sample_every_n_steps=int(state['sample_every_n_steps'].value or 0),
                                seed=42,
                                workers=2,
                                learning_rate=state['learning_rate'].value or '8e-4',
                                network_dim=int(state['network_dim'].value or 4),
                                network_alpha=float(state['network_alpha'].value or 1.0),
                                guidance_scale=1.0,
                                timestep_sampling='shift',
                                advanced_flags=[],
                                uploaded_files=uploads,
                                caption_rows=state['caption_rows'],
                                log_fn=log_fn,
                            )
                            
                            # Update script and config
                            if current_slot:
                                with current_slot:
                                    try:
                                        script_content = open(script_path, 'r', encoding='utf-8').read()
                                        ui_set_script(script_content)
                                    except Exception:
                                        pass
                                    try:
                                        config_content = open(toml_path, 'r', encoding='utf-8').read()
                                        ui_set_config(config_content)
                                    except Exception:
                                        pass
                            
                            # Launch process
                            import subprocess
                            cmd = script_path if platform.system().lower().startswith('win') else f"bash \"{script_path}\""
                            if current_slot:
                                with current_slot:
                                    ui_log(f"[LAUNCH] Starting training with command: {cmd}")
                                    ui_log(f"[DEBUG] Working directory: {ROOT}")
                                    ui_log(f"[DEBUG] Script exists: {os.path.exists(script_path)}")
                                    ui_log(f"[DEBUG] Dataset exists: {os.path.exists(os.path.join(ROOT, 'datasets', 'uygug'))}")
                            
                            # Set UTF-8 environment for subprocess
                            env = os.environ.copy()
                            env['PYTHONIOENCODING'] = 'utf-8:replace'
                            env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
                            env['PYTHONUTF8'] = '1'
                            
                            # Use simpler process launch without redirection for now
                            proc = subprocess.Popen(
                                cmd,
                                cwd=ROOT,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                env=env,
                                encoding='utf-8',
                                errors='replace',
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system().lower().startswith('win') else 0
                            )
                            state['proc_handle'] = proc
                            global CURRENT_PROC
                            CURRENT_PROC = proc
                            
                            # Read output with termination checking
                            while True:
                                try:
                                    line = proc.stdout.readline()
                                    if not line:  # Process ended
                                        break
                                    if state.get('proc_handle') is None:  # Process was stopped
                                        if current_slot:
                                            with current_slot:
                                                ui_log("[STOP] Process termination detected")
                                        break
                                    if current_slot:
                                        with current_slot:
                                            ui_log(line.rstrip('\n'))
                                except UnicodeDecodeError:
                                    # Skip problematic lines instead of crashing
                                    if current_slot:
                                        with current_slot:
                                            ui_log("[ENCODING] Skipped line with encoding issues")
                                    continue
                                except Exception as e:
                                    if current_slot:
                                        with current_slot:
                                            ui_log(f"[OUTPUT ERROR] {e}")
                                    break
                            
                            # Wait for process to finish
                            proc.wait()
                            return_code = proc.returncode
                            if current_slot:
                                with current_slot:
                                    if return_code == 0:
                                        ui_log("[COMPLETE] Training completed successfully")
                                    else:
                                        ui_log(f"[COMPLETE] Training ended with code {return_code}")
                                        
                        except Exception as e:
                            if current_slot:
                                with current_slot:
                                    ui_log(f"[ERROR] {e}")
                        finally:
                            state['proc_handle'] = None
                            CURRENT_PROC = None
                            # Reset button state
                            if current_slot:
                                with current_slot:
                                    set_btn_running(False)
                                    set_training_flag(False)
                            else:
                                try:
                                    state['train_btn'].text = 'Start training'
                                    state['train_btn'].props('color=positive')
                                except Exception:
                                    pass

                    import threading as _threading
                    _threading.Thread(target=worker, daemon=True).start()

                state['train_btn'].on('click', lambda _: toggle_training())

        # cross-platform stop logic
        def stop_process(proc):
            try:
                if proc is None:
                    return
                if platform.system().lower().startswith('win'):
                    # On Windows, try multiple approaches
                    try:
                        import subprocess
                        # Kill the process tree using taskkill
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], 
                                     capture_output=True, timeout=5)
                    except Exception:
                        pass
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        pass
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.kill()
                    except Exception:
                        pass
                else:
                    # On Unix-like systems, kill the process group
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        import time
                        time.sleep(1)  # Give it a moment to terminate gracefully
                        if proc.poll() is None:  # Still running
                            os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        # Fallback to direct process termination
                        try:
                            proc.terminate()
                            import time
                            time.sleep(1)
                            if proc.poll() is None:
                                proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass

        # Bottom: Terminal full width - resizable
        with ui.row().classes('w-full').style('margin-top: 12px; width: 100%'):
            with ui.card().classes('w-full p-2').style('min-height: 50vh; height: 50vh; display: flex; flex-direction: column; resize: vertical; overflow: auto'):
                ui.markdown('### Terminal')
                state['terminal_log'] = ui.log(max_lines=50000).classes('w-full').style('flex: 1 1 auto; background:#0b0b0b; color:#e6e6e6; font-family:monospace;')
                ui.button('Clear', on_click=lambda: setattr(state['terminal_log'], 'lines', [])).props('outline size=sm')


def page_metrics() -> None:
    ui.markdown('### Metrics')
    from tensorboard.backend.event_processing import event_accumulator  # type: ignore

    runs = list_runs()
    state: Dict[str, Any] = {}
    
    def refresh_runs_list():
        """Refresh the runs dropdown with current training runs"""
        runs = list_runs()
        state['primary'].options = runs
        state['overlays'].options = runs
        if runs and not state['primary'].value:
            state['primary'].value = runs[0]
        return runs
    
    with ui.row():
        state['primary'] = ui.select(options=runs, label='Primary run', value=(runs[0] if runs else None))
        state['overlays'] = ui.select(options=runs, label='Overlay runs (optional)', multiple=True)
        state['autorefresh'] = ui.toggle(['Auto-refresh'], value=[True]).props('keep-color')
        state['interval'] = ui.number(label='Interval (seconds)', value=2, format='%.0f')
        refresh_btn = ui.button('Refresh Now')
        refresh_runs_btn = ui.button('Refresh Runs List', on_click=lambda: refresh_runs_list())

    ui.separator()
    ui.markdown('#### Health & Progress')
    with ui.row():
        state['health'] = ui.label('No training issues detected').classes('text-green-600')
        ui.space()
        state['progress'] = ui.label('Progress: 0/0')
        state['curr_loss'] = ui.label('Current Loss: -')
        state['curr_lr'] = ui.label('Learning Rate: -')

    ui.separator()
    ui.markdown('#### Loss / Learning Rate')
    loss_chart = ui.echart({'xAxis': {'type': 'category', 'data': []},
                            'yAxis': {'type': 'value'},
                            'series': []}).classes('w-full h-64')
    lr_chart = ui.echart({'xAxis': {'type': 'category', 'data': []},
                          'yAxis': {'type': 'value'},
                          'series': []}).classes('w-full h-48')

    ui.separator()
    ui.markdown('#### Recent Events')
    events_table = ui.table(columns=[{'name':'time','label':'Time','field':'time'},
                                     {'name':'event','label':'Event','field':'event'},
                                     {'name':'step','label':'Step','field':'step'},
                                     {'name':'details','label':'Details','field':'details'}],
                            rows=[]).props('dense flat')

    ui.separator()
    ui.markdown('#### Samples Timeline')
    samples_row = ui.row().classes('w-full flex-wrap gap-2')
    prompt_selector = ui.select(options=[], label='Prompt index')

    def load_tb_scalars(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
        try:
            ev = event_accumulator.EventAccumulator(log_dir)
            ev.Reload()
            scalars: Dict[str, List[Tuple[int, float]]] = {}
            for tag in ev.Tags().get('scalars', []):
                data = [(e.step, e.value) for e in ev.Scalars(tag)]
                scalars[tag] = data
            return scalars
        except Exception:
            return {}

    def update_metrics() -> None:
        # Auto-refresh runs list to catch new training runs
        refresh_runs_list()
        
        primary = state['primary'].value
        overlays = state['overlays'].value or []
        all_runs = [primary] + [r for r in overlays if r and r != primary]
        series_loss = []
        series_lr = []
        x_axis = []
        
        for idx, run_path in enumerate([r for r in all_runs if r]):
            logs_dir = os.path.join(run_path, 'logs')
            scalars = load_tb_scalars(logs_dir)
            
            # Try multiple loss tag names
            loss = scalars.get('loss') or scalars.get('train/loss') or scalars.get('training_loss') or []
            lr = scalars.get('lr') or scalars.get('learning_rate') or scalars.get('train/lr') or []
            
            steps = [s for s, _ in loss]
            losses = [v for _, v in loss]
            lrs = [v for _, v in lr]
            if steps and not x_axis:
                x_axis = steps
            series_loss.append({'type': 'line', 'name': os.path.basename(run_path), 'data': losses})
            if lrs:
                series_lr.append({'type': 'line', 'name': os.path.basename(run_path), 'data': lrs})

            # update header
            if idx == 0 and steps:
                state['progress'].text = f'Progress: {steps[-1]} steps'
                if losses:
                    state['curr_loss'].text = f'Current Loss: {losses[-1]:.4f}'
                if lrs:
                    state['curr_lr'].text = f'Learning Rate: {lrs[-1]:.2e}'
                    
            # Update health status
            if idx == 0:
                if scalars:
                    available_tags = list(scalars.keys())
                    state['health'].text = f'Tracking: {", ".join(available_tags[:3])}'
                    state['health'].classes('text-green-600')
                else:
                    logs_exists = os.path.exists(logs_dir)
                    state['health'].text = f'Logs dir exists: {logs_exists} | Path: {logs_dir}'
                    state['health'].classes('text-amber-600')

            # samples per prompt index
            try:
                samples_dir = os.path.join(run_path, 'sample')
                # discover prompt indices from filenames pattern *_stepIndex_promptIndex_timestamp.png
                files = []
                if os.path.isdir(samples_dir):
                    files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
                # map prompt index → list of files
                index_to_files: Dict[int, List[str]] = {}
                for name in sorted(files):
                    parts = name.split('_')
                    if len(parts) >= 3 and parts[-1].lower().endswith('.png'):
                        try:
                            step = int(parts[-3])  # best-effort; schema may differ
                            prompt_idx = int(parts[-2])
                        except Exception:
                            continue
                        index_to_files.setdefault(prompt_idx, []).append(os.path.join(samples_dir, name))
                if index_to_files:
                    prompt_selector.options = [str(k) for k in sorted(index_to_files.keys())]
                    sel = int(prompt_selector.value or list(sorted(index_to_files.keys()))[0])
                    thumbs = index_to_files.get(sel, [])[-12:]
                    with samples_row:
                        samples_row.clear()
                        for p in thumbs:
                            ui.image(p).classes('w-40 h-40 object-cover')
            except Exception:
                pass

        loss_chart.options['xAxis']['data'] = x_axis
        loss_chart.options['series'] = series_loss
        loss_chart.update()
        lr_chart.options['xAxis']['data'] = x_axis
        lr_chart.options['series'] = series_lr
        lr_chart.update()

    refresh_btn.on('click', lambda e: update_metrics())

    import time
    last_tick = {'t': 0.0}

    def periodic() -> None:
        try:
            if not state['autorefresh'].value:
                return
            now = time.time()
            interval = int(state['interval'].value or 2)
            if now - last_tick['t'] >= interval:
                update_metrics()
                last_tick['t'] = now
        except Exception:
            pass

    ui.timer(1.0, periodic)


def page_publish() -> None:
    ui.markdown('### Publish')
    ui.button('Open Gradio Publish', on_click=open_gradio_ui)


@ui.page('/')
def main_page() -> None:
    top_navbar()
    with ui.tabs().classes('w-full') as tabs:
        train_tab = ui.tab('Train')
        metrics_tab = ui.tab('Metrics')
        publish_tab = ui.tab('Publish')
    with ui.tab_panels(tabs, value=train_tab).classes('w-full'):
        with ui.tab_panel(train_tab):
            page_train()
        with ui.tab_panel(metrics_tab):
            page_metrics()
        with ui.tab_panel(publish_tab):
            page_publish()


def run() -> None:
    ui.run(reload=False, title='FluxGym New UI', port=7861)


if __name__ == '__main__':
    run()


def prompt_reset_if_dirty() -> None:
    # Dirty if any primary inputs are non-empty or files uploaded
    # This is a lightweight client-side warning; does not block server state
    def open_dialog():
        with ui.dialog() as dlg, ui.card():
            ui.markdown('You have unsaved state. Reloading will reset the UI.')
            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=dlg.close).props('outline')
                def do_reload():
                    dlg.close()
                    ui.run_javascript('location.reload()')
                ui.button('Reload', on_click=do_reload).props('color=primary')
        dlg.open()
    open_dialog()


@nicegui_app.get('/stop_training')
async def stop_training_endpoint():
    global CURRENT_PROC
    try:
        if CURRENT_PROC is not None:
            stop_process(CURRENT_PROC)
            CURRENT_PROC = None
        return {'status': 'ok'}
    except Exception:
        return {'status': 'error'}


