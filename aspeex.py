import wave
from speex import WBDecoder

def usize(buffer):
    sizes = []
    for char in buffer:
        sizes.append(char & 0x7f)
        if char & 0x80 == 0:
            break

    size = 0
    for i in range(len(sizes)):
        size += sizes[i] * (2**7)**(len(sizes) - i - 1)
    return len(sizes), size


async def speex2wav(in_file,out_file):
    vocoded = open(in_file, 'rb').read() if type(in_file)==str else in_file # speex binary
    decoder = WBDecoder()
    i = 0
    pcm = b''
    while i < len(vocoded):
        header_size, packet_size = usize(vocoded[i:])
        pcm += decoder.decode(vocoded[i + header_size:i + header_size + packet_size])
        i += header_size + packet_size
    wavfile = wave.open(out_file, 'wb')
    wavfile.setnchannels(1)
    wavfile.setsampwidth(16 // 8)
    wavfile.setframerate(16000)
    wavfile.writeframes(pcm)
    wavfile.close()

