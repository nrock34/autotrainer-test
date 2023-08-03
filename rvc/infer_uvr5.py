import os, sys, torch, warnings, pdb
import math

now_dir = os.getcwd()
sys.path.append(now_dir)
from json import load as ll

warnings.filterwarnings("ignore")
import librosa
import importlib
import numpy as np
import hashlib, math
from tqdm import tqdm
from rvc.lib.uvr5_pack.lib_v5 import spec_utils
from rvc.lib.uvr5_pack.utils import _get_name_params, inference
from rvc.lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
import soundfile as sf
from rvc.lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from rvc.lib.uvr5_pack.lib_v5 import nets as nets


class _audio_pre_:
    def __init__(self, agg, model_path, device, is_half):
        self.model_path = model_path
        self.device = device
        self.agg = agg
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        self.window_size = 512
        
        mp = ModelParameters("/notebooks/autotrainer/rvc/lib/uvr5_pack/lib_v5/modelparams/4band_44100.json") #+ ("4band_v2.json" if "9_HP2-UVR" not in model_path else "4band_44100.json"))
        nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(model_path).st_size / 1024)
        nn_arch_size = 537238
        print(nn_arch_size)
        
        if nn_arch_size in vr_5_1_models:
            #model = nets_new.CascadedNet(mp.param['bins'] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            pass
        else:
            print("@model")
            model = nets.determine_model_capacity(mp.param['bins'] * 2, nn_arch_size)
        print("@cpk")    
        cpk = torch.load(model_path, map_location="cpu")
        print("@load model")
        model.load_state_dict(cpk)
        print("@eval")
        #model.eval()
        
        #if is_half:
        #    model = model.half().to(device)
        #else:
        model = model.to(device)
        
        
        self.mp = mp
        self.model = model
        self.input_high_end_h = None
        self.batch_size = 4
        self.music_file = None
        
        self.aggressiveness = {'value': agg, 
                               'split_bin': self.mp.param['band'][1]['crop_stop'], 
                               'aggr_correction': self.mp.param.get('aggr_correction')}

    def seperate(self):
        if True:
            #self.start_inference_console_write()
            if True:
                if OPERATING_SYSTEM == 'Darwin':
                    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                else:
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            else:
                device = torch.device('cpu')

            nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

            #if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            #    self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            if True:
                model = nets.determine_model_capacity(mp.param['bins'] * 2, nn_arch_size)
                            
            model.load_state_dict(torch.load(model_path, map_location=cpu)) 
            model.to(device) 

            #self.running_inference_console_write()
                        
            y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
            #self.write_to_console(DONE, base_text='')
            print("DONE")
            
        '''   
        if self.is_secondary_model_activated:
            if self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)
        

        if not self.is_secondary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.primary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = spec_utils.normalize(self.spec_to_wav(y_spec), self.is_normalization).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = {self.primary_stem: self.primary_source}
            
            self.write_audio(primary_stem_path, self.primary_source, 44100, self.secondary_source_primary)

        if not self.is_primary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.secondary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec)
                self.secondary_source = spec_utils.normalize(self.spec_to_wav(v_spec), self.is_normalization).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = {self.secondary_stem: self.secondary_source}
            
            self.write_audio(secondary_stem_path, self.secondary_source, 44100, self.secondary_source_secondary)
        '''    

        torch.cuda.empty_cache()
        #secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        #self.cache_source(secondary_sources)

        #if self.is_secondary_model:
        #   return secondary_sources
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            #if OPERATING_SYSTEM == 'Darwin':
            #   wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            #else:
            wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(self.music_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                    
                if not np.any(X_wave[d]) and self.music_file.endswith('.mp3'):
                    X_wave[d] = rerun_mp3(self.music_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'], 
                                                            self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            
            '''
            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]
            '''
            
        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        
        del X_wave, X_spec_s

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model.offset) // roi_size
            total_iterations = patches//self.batch_size #if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    #self.progress_value += 1
                    #if self.progress_value >= total_iterations:
                    #    self.progress_value = total_iterations
                    #self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        #raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                        pass
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    #raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pass
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            
            #is_non_accom_stem = False
            #for stem in NON_ACCOM_STEMS:
            #    if stem == self.primary_stem:
            is_non_accom_stem = False
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            #if self.is_post_process:
            #    mask = spec_utils.merge_artifacts(mask, thres=0.2)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        

        if True is False:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        
        if True:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp)
            
        return wav


    def _path_audio_(self, music_file, ins_root=None, vocal_root=None, format="flac"):

        self.music_file = music_file

        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs('./ins_root', exist_ok=True)
            ins_root = './ins_root'
        if vocal_root is not None:
            os.makedirs('./vocal_root', exist_ok=True)
            vocal_root = './vocal_root'

        if True:
            #self.start_inference_console_write()
            if True:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            else:
                device = torch.device('cpu')

        '''
            nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

            #if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            #    self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
            if True:
                model = nets.determine_model_capacity(mp.param['bins'] * 2, nn_arch_size)
                            
            model.load_state_dict(torch.load(model_path, map_location=cpu)) 
            model.to(device) 

            #self.running_inference_console_write()
        '''
                        
        y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
        #self.write_to_console(DONE, base_text='')
        print("DONE")
        torch.cuda.empty_cache()

        '''
        if ins_root is None and vocal_root is None:
            return "No save root."
        
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                (
                    X_wave[d],
                    _,
                ) = librosa.core.load(  # 理论上librosa读取可能对某些音频有bug，应该上ffmpeg读取，但是太麻烦了弃坑
                    music_file,
                    bp["sr"],
                    False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    self.mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        '''

        if ins_root is not None:
            print("%s instruments done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        ins_root,
                        "instrument_{}.{}".format(name, format),
                    ),
                    (np.array(self.spec_to_wav(y_spec)) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )  #
            else:
                path = os.path.join(
                    ins_root, "instrument_{}.wav".format(name)
                )
                sf.write(
                    path,
                    (np.array(self.spec_to_wav(y_spec)) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    os.system(
                        "ffmpeg -i %s -vn %s -q:a 2 -y"
                        % (path, path[:-4] + ".%s" % format)
                    )
        if vocal_root is not None:
            print("%s vocals done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        vocal_root,
                        "vocal_{}.{}".format(name, format),
                    ),
                    (np.array(self.pec_to_wav(v_spec)) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
            else:
                path = os.path.join(
                    vocal_root, "vocal_{}.wav".format(name)
                )
                sf.write(
                    path,
                    (np.array(self.pec_to_wav(v_spec)) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    os.system(
                        "ffmpeg -i %s -vn %s -q:a 2 -y"
                        % (path, path[:-4] + ".%s" % format)
                    )


class _audio_pre_new:
    def __init__(self, agg, model_path, device, is_half):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("/notebooks/autotrainer/rvc/lib/uvr5_pack/lib_v5/modelparams/4band_v3.json")
        nout = 64 if "DeReverb" in model_path else 48
        model = CascadedNet(mp.param["bins"] * 2, nout)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(
        self, music_file, vocal_root=None, ins_root=None, format="flac"
    ):  # 3个VR模型vocal和ins是反的
        if ins_root is None and vocal_root is None:
            return "No save root."
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs("./echo_ins_root", exist_ok=True)
            ins_root = "./echo_ins_root"
        if vocal_root is not None:
            os.makedirs("./echo_vocal_root", exist_ok=True)
            vocal_root = "./echo_vocal_root"
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                (
                    X_wave[d],
                    _,
                ) = librosa.core.load(  # 理论上librosa读取可能对某些音频有bug，应该上ffmpeg读取，但是太麻烦了弃坑
                    music_file,
                    bp["sr"],
                    False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    self.mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        if ins_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(
                    self.data["high_end_process"], y_spec_m, input_high_end, self.mp
                )
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                    y_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            print("%s instruments done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        ins_root,
                        "instrument_{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (np.array(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )  #
            else:
                path = os.path.join(
                    ins_root, "instrument_{}_{}.wav".format(name, self.data["agg"])
                )
                sf.write(
                    path,
                    (np.array(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    os.system(
                        "ffmpeg -i %s -vn %s -q:a 2 -y"
                        % (path, path[:-4] + ".%s" % format)
                    )
        if vocal_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(
                    self.data["high_end_process"], v_spec_m, input_high_end, self.mp
                )
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                    v_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            print("%s vocals done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        vocal_root,
                        "vocal_{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (np.array(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
            else:
                path = os.path.join(
                    vocal_root, "vocal_{}_{}.wav".format(name, self.data["agg"])
                )
                sf.write(
                    path,
                    (np.array(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    os.system(
                        "ffmpeg -i %s -vn %s -q:a 2 -y"
                        % (path, path[:-4] + ".%s" % format)
                    )


if __name__ == "__main__":
    device = "cuda"
    is_half = True
    # model_path = "uvr5_weights/2_HP-UVR.pth"
    # model_path = "uvr5_weights/VR-DeEchoDeReverb.pth"
    # model_path = "uvr5_weights/VR-DeEchoNormal.pth"
    model_path = "uvr5_weights/DeEchoNormal.pth"
    # pre_fun = _audio_pre_(model_path=model_path, device=device, is_half=True,agg=10)
    pre_fun = _audio_pre_new(model_path=model_path, device=device, is_half=True, agg=10)
    audio_path = "雪雪伴奏对消HP5.wav"
    save_path = "opt"
    pre_fun._path_audio_(audio_path, save_path, save_path)
