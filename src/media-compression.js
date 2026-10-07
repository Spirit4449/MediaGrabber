const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawn } = require('node:child_process');

function run(command, args, timeoutMs = 30 * 60 * 1000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-8192); });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout);
      else reject(new Error(`${command} exited ${code}: ${stderr}`));
    });
  });
}
function fraction(value) {
  const [a, b] = String(value || '1:1').split(':').map(Number);
  return a > 0 && b > 0 ? a / b : 1;
}
function geometry(video) {
  const rotation = Number(video.side_data_list?.find(item => item.rotation !== undefined)?.rotation || video.tags?.rotate || 0);
  let width = video.width * fraction(video.sample_aspect_ratio), height = video.height;
  if (Math.abs(rotation) % 180 === 90) [width, height] = [height, width];
  return { width, height, ratio: width / height };
}
function dimensions(video, videoKbps) {
  const display = geometry(video);
  const shortEdge = videoKbps >= 1400 ? 720 : videoKbps >= 650 ? 480 : 360;
  const longEdge = Math.round(shortEdge * 16 / 9);
  const landscape = display.width >= display.height;
  const factor = Math.min(1, (landscape ? longEdge : shortEdge) / display.width, (landscape ? shortEdge : longEdge) / display.height);
  return { width: Math.max(2, Math.round(display.width * factor / 2) * 2), height: Math.max(2, Math.round(display.height * factor / 2) * 2) };
}
function createCompressor({ ffmpeg = 'ffmpeg', ffprobe = 'ffprobe', targetMb = 47, audioKbps = 192, minVideoKbps = 300, maxVideoKbps = 2500, preset = 'medium', timeoutMs = 1800000 } = {}) {
  for (const [name, value] of Object.entries({ targetMb, audioKbps, minVideoKbps, maxVideoKbps, timeoutMs })) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
  }
  let queue = Promise.resolve();
  async function probe(file) {
    return JSON.parse(await run(ffprobe, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', file], 30000));
  }
  async function encode(file, { isVideo, isAudio, maxBytes }) {
    const work = await fs.mkdtemp(path.join(os.tmpdir(), 'mediagrabber-'));
    const output = path.join(work, isVideo ? 'preview.mp4' : 'preview.mp3');
    try {
      const metadata = await probe(file);
      const duration = Number(metadata.format.duration);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error('Invalid media duration');
      const video = metadata.streams.find(stream => stream.codec_type === 'video' && !stream.disposition?.attached_pic);
      const audio = metadata.streams.find(stream => stream.codec_type === 'audio');
      const target = Math.floor(Math.min(maxBytes * 0.94, targetMb * 1024 * 1024));
      const totalKbps = Math.floor(target * 8 / duration / 1000);
      if (totalKbps <= 0) throw new Error('No usable bitrate budget');
      if (isVideo && video) {
        const soundKbps = audio ? Math.min(128, Math.max(64, Math.floor(totalKbps * 0.1))) : 0;
        for (let attempt = 0; attempt < 2; attempt++) {
          const bitrate = Math.floor(Math.min(maxVideoKbps, (totalKbps - soundKbps) * (attempt ? 0.85 : 1)));
          if (bitrate < minVideoKbps) throw new Error('Video would require excessive quality loss; use the original link');
          const size = dimensions(video, bitrate);
          // FFmpeg autorotates on decode. Dimensions include display SAR; setsar is safe after this resize.
          const common = ['-y', '-v', 'error', '-i', file, '-map', `0:${video.index}`, '-vf', `scale=${size.width}:${size.height}:flags=lanczos,setsar=1`, '-c:v', 'libx264', '-preset', preset, '-pix_fmt', 'yuv420p', '-b:v', `${bitrate}k`, '-passlogfile', path.join(work, 'pass'), '-threads', '2'];
          await run(ffmpeg, [...common, '-pass', '1', '-an', '-f', 'null', '-'], timeoutMs);
          await run(ffmpeg, [...common, ...(audio ? ['-map', `0:${audio.index}`, '-c:a', 'aac', '-b:a', `${soundKbps}k`] : ['-an']), '-pass', '2', '-map_metadata', '-1', '-metadata:s:v:0', 'rotate=0', '-movflags', '+faststart', output], timeoutMs);
          const stat = await fs.stat(output);
          if (stat.size > maxBytes) continue;
          const verified = await probe(output);
          const result = verified.streams.find(stream => stream.codec_type === 'video');
          if (Math.abs(Number(verified.format.duration) - duration) > Math.max(1, duration * 0.01)) throw new Error('Compressed duration changed unexpectedly');
          if (Math.abs(geometry(result).ratio / geometry(video).ratio - 1) > 0.01) throw new Error('Compressed display aspect ratio changed');
          return { path: output, ext: '.mp4', width: result.width, height: result.height, duration, cleanup: () => fs.rm(work, { recursive: true, force: true }) };
        }
        throw new Error('Compressed video still exceeds Telegram limit');
      }
      if (!isAudio || !audio) throw new Error('No supported media stream');
      const bitrate = Math.floor(Math.min(audioKbps, totalKbps));
      if (bitrate < 48) throw new Error('Audio would require excessive quality loss; use the original link');
      await run(ffmpeg, ['-y', '-v', 'error', '-i', file, '-map', `0:${audio.index}`, '-vn', '-c:a', 'libmp3lame', '-b:a', `${bitrate}k`, output], timeoutMs);
      if ((await fs.stat(output)).size > maxBytes) throw new Error('Compressed audio still exceeds Telegram limit');
      return { path: output, ext: '.mp3', cleanup: () => fs.rm(work, { recursive: true, force: true }) };
    } catch (error) {
      await fs.rm(work, { recursive: true, force: true });
      return { error: error.message };
    }
  }
  function compress(file, options) {
    const job = queue.then(() => encode(file, options));
    queue = job.catch(() => {});
    return job;
  }
  return { compress, probe };
}
module.exports = { createCompressor, dimensions, geometry, run };
