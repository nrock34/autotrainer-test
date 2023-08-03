import sys
import os
from sp2mp3 import *
#import spotify
import ffmpeg
import traceback, pdb
import sp2mp3 as sp
from rvc.infer_uvr5 import _audio_pre_ as AudioProcessor
from rvc.infer_uvr5 import _audio_pre_new as EchoAudioProcessor
from rvc.MDXNet import MDXNetDereverb as ReverbAudioProcessor

#infos = []

gatherMusic = False
musicFolder = ""
trgtFolder = "drake"

SPOTIFY_API_CLIENT_ID = 'e5770798a03549bd94033e657956c063'
SPOTIFY_API_CLIENT_SECRET = 'de883b0639b045a0a6e2032565b420e1'
SPOTIFY_USERNAME = "j2epic"

PLAYLIST_URI = '18fPAYPe3ZsA708Okm21EH'

tpath = 'opt'


if gatherMusic:
    #musicFolder = sp.sp2mp3(SPOTIFY_API_CLIENT_ID, SPOTIFY_API_CLIENT_SECRET, SPOTIFY_USERNAME, PLAYLIST_URI)
    os.chdir("/notebooks/autotrainer/")
    now_dir = os.getcwd()
    sys.path.append(now_dir)
    print(os.getcwd())
    tmp = os.path.join(now_dir, "TEMP")
    print(tmp)
    trgtFolder = musicFolder
else:
    os.chdir("/notebooks/autotrainer/")
    now_dir = os.getcwd()
    sys.path.append(now_dir)
    tmp = os.path.join(now_dir, "TEMP")
    

voc_root = "vocal_root"
echo_root = "echo_vocal_root"
dereveb_root = "dereverb_vocal_root"


for file in os.listdir("/notebooks/autotrainer/" + trgtFolder + '/'):
    newfile=""
    for letter in file:
        newfile += letter if " " not in letter else "-"
        #print(newfile)
    os.rename('drake/' + file, 'drake/' + newfile)
    

device = 'cuda'
isHalf = True


vocSplitter = AudioProcessor(model_path="/notebooks/autotrainer/rvc/uvr5_weights/9_HP2-UVR.pth", device = device, is_half = isHalf, agg=10)
#vocSplitter._path_audio_(input, tpath, tpath, 'wav')
deReverber = ReverbAudioProcessor(18)
#deReverber._path_audio_(input)
deEchoer = EchoAudioProcessor(model_path="/notebooks/autotrainer/rvc/uvr5_weights/VR-DeEchoAggressive.pth", device = device, is_half = isHalf, agg=10)
#deEchoer._path_audio_(input, tpath, tpath, 'wav')


def main():
    infos = []
    #print("HI5")
    for file in os.listdir('drake/'):
        #print("HI")
        print(file)
        if 'mp3' in file:
            song = os.path.join('drake/', file)
            print("")
        else:
            continue

        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(song, cmd="ffprobe")
            if (
                info["streams"][0]["channels"] == 2
                and info["streams"][0]["sample_rate"] == "44100"
            ):
                need_reformat = 0
                vocSplitter._path_audio_(song, tpath, tpath, 'wav')
                done = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(song))
            os.system(
                "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                % (song, tmp_path)
            )
            song = tmp_path
        try:
            if done == 0:
                vocSplitter._path_audio_(song, tpath, tpath, 'wav')
            print("%s->Success" % (os.path.basename(song)))
            yield "\n".join(infos)
        except:
            infos.append("%s->%s" % (os.path.basename(song), traceback.format_exc()))
            yield "\n".join(infos)

        #vocSplitter._path_audio_(song, tpath, tpath, 'wav')

    for file in os.listdir(voc_root):
        song = os.path.join(voc_root, file)

        deReverber._path_audio_(song)

    for file in os.listdir(dereveb_root):
        song = os.path.join(dereveb_root, file)

        deEchoer._path_audio_(song, opt, opt, 'wav')


for i in main():
    print(i, end=" ")

"""
f = ''
for file in os.listdir('drake/'):
    if '.mp3' in file:
        f = file
        break

tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(os.path.join('drake/', file)))
print(tmp_path)
"""


print('done')







