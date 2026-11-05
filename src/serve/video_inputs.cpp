#include "serve/video_inputs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

#include "common/base64.hpp"

#ifdef DGPP_ENABLE_VIDEO
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#endif

namespace dgpp::serve {
bool native_video_available() {
#ifdef DGPP_ENABLE_VIDEO
  return true;
#else
  return false;
#endif
}

#ifdef DGPP_ENABLE_VIDEO
namespace {
constexpr int kMaxDecodedFrames = 100000;
constexpr int64_t kMaxDecodedPixels = 32ll * 1024 * 1024;
void check_av(int rc, const char* operation) {
  if (rc >= 0) return;
  char message[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(rc, message, sizeof(message));
  throw std::invalid_argument(std::string(operation) + ": " + message);
}
// Custom AVIO owns only the supplied bytes. No filename, subprocess or
// network protocol is needed to decode an admission record.
struct Decoder {
  std::string_view bytes;
  int64_t position = 0;
  AVIOContext* io = nullptr;
  AVFormatContext* format = nullptr;
  AVCodecContext* codec = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* frame = nullptr;
  SwsContext* converter = nullptr;
  int stream = -1, width = 0, height = 0;
  double fps = 0;
  static int read(void* opaque, uint8_t* dst, int count) {
    auto& self = *static_cast<Decoder*>(opaque);
    const size_t n = std::min(size_t(count), self.bytes.size() - size_t(self.position));
    if (!n) return AVERROR_EOF;
    std::memcpy(dst, self.bytes.data() + self.position, n);
    self.position += n;
    return static_cast<int>(n);
  }
  static int64_t seek(void* opaque, int64_t offset, int whence) {
    auto& self = *static_cast<Decoder*>(opaque);
    if (whence == AVSEEK_SIZE) return self.bytes.size();
    whence &= ~AVSEEK_FORCE;
    const int64_t base = whence == SEEK_SET   ? 0
                         : whence == SEEK_CUR ? self.position
                         : whence == SEEK_END ? int64_t(self.bytes.size())
                                              : -1;
    if (base < 0 || offset < -base || offset > int64_t(self.bytes.size()) - base)
      return AVERROR(EINVAL);
    return self.position = base + offset;
  }
  void close() {
    sws_freeContext(converter);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io) av_freep(&io->buffer);
    avio_context_free(&io);
  }
  explicit Decoder(std::string_view data) : bytes(data) {
    try {
      auto* buffer = static_cast<uint8_t*>(av_malloc(32768));
      if (!buffer) throw std::bad_alloc();
      io = avio_alloc_context(buffer, 32768, 0, this, read, nullptr, seek);
      if (!io) {
        av_free(buffer);
        throw std::bad_alloc();
      }
      format = avformat_alloc_context();
      if (!format) throw std::bad_alloc();
      format->pb = io;
      format->flags |= AVFMT_FLAG_CUSTOM_IO;
      format->max_streams = 8;
      format->probesize = 1 << 20;
      format->max_analyze_duration = 5 * AV_TIME_BASE;
      AVDictionary* options = nullptr;
      av_dict_set(&options, "format_whitelist", "mov,matroska,webm", 0);
      av_dict_set(&options, "protocol_whitelist", "", 0);
      av_dict_set(&options, "enable_drefs", "0", 0);
      const int opened = avformat_open_input(&format, nullptr, nullptr, &options);
      av_dict_free(&options);
      check_av(opened, "cannot open video");
      check_av(avformat_find_stream_info(format, nullptr), "cannot inspect video");
      const AVCodec* decoder = nullptr;
      stream = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
      check_av(stream, "video stream is missing");
      codec = avcodec_alloc_context3(decoder);
      if (!codec) throw std::bad_alloc();
      check_av(avcodec_parameters_to_context(codec, format->streams[stream]->codecpar),
               "invalid video parameters");
      codec->thread_count = 2;
      codec->max_pixels = kMaxDecodedPixels;
      check_av(avcodec_open2(codec, decoder, nullptr), "cannot open video decoder");
      fps = av_q2d(av_guess_frame_rate(format, format->streams[stream], nullptr));
      if (!std::isfinite(fps) || fps <= 0 || fps > 1000)
        throw std::invalid_argument("video has no usable frame rate");
      packet = av_packet_alloc();
      frame = av_frame_alloc();
      if (!packet || !frame) throw std::bad_alloc();
    } catch (...) {
      close();
      throw;
    }
  }
  ~Decoder() { close(); }
  void decode(const std::function<void(int, double)>& consume) {
    int count = 0;
    double first = 0, previous = -1;
    const auto drain = [&] {
      while (true) {
        const int rc = avcodec_receive_frame(codec, frame);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        check_av(rc, "cannot decode video frame");
        if (count == kMaxDecodedFrames)
          throw std::invalid_argument("video exceeds 100000 decoded frames");
        if (frame->width < 1 || frame->height < 1 || frame->width > 16384 ||
            frame->height > 16384 || int64_t(frame->width) * frame->height > kMaxDecodedPixels)
          throw std::invalid_argument("video frame exceeds the dimension limit");
        if (!count) {
          width = frame->width;
          height = frame->height;
        }
        if (frame->width != width || frame->height != height)
          throw std::invalid_argument("video changes frame dimensions");
        double time =
            frame->best_effort_timestamp == AV_NOPTS_VALUE
                ? count / fps
                : frame->best_effort_timestamp * av_q2d(format->streams[stream]->time_base);
        if (!count) first = time;
        time -= first;
        if (!std::isfinite(time) || time < previous || time > 86400)
          throw std::invalid_argument("video timestamps are invalid or exceed 24 hours");
        previous = time;
        consume(count++, time);
        av_frame_unref(frame);
      }
    };
    while (true) {
      const int rc = av_read_frame(format, packet);
      if (rc == AVERROR_EOF) break;
      check_av(rc, "cannot read video packet");
      if (packet->stream_index == stream) {
        check_av(avcodec_send_packet(codec, packet), "cannot submit video packet");
        drain();
      }
      av_packet_unref(packet);
    }
    check_av(avcodec_send_packet(codec, nullptr), "cannot finish video");
    drain();
    if (!count) throw std::invalid_argument("video contains no frames");
  }
  std::vector<uint8_t> rgb() {
    converter = sws_getCachedContext(converter, width, height, AVPixelFormat(frame->format), width,
                                     height, AV_PIX_FMT_RGB24, SWS_BICUBIC | SWS_ACCURATE_RND,
                                     nullptr, nullptr, nullptr);
    if (!converter) throw std::invalid_argument("unsupported video pixel format");
    int space = SWS_CS_DEFAULT;
    if (frame->colorspace == AVCOL_SPC_BT709)
      space = SWS_CS_ITU709;
    else if (frame->colorspace == AVCOL_SPC_FCC)
      space = SWS_CS_FCC;
    else if (frame->colorspace == AVCOL_SPC_SMPTE240M)
      space = SWS_CS_SMPTE240M;
    else if (frame->colorspace == AVCOL_SPC_BT2020_NCL || frame->colorspace == AVCOL_SPC_BT2020_CL)
      space = SWS_CS_BT2020;
    const int* coefficients = sws_getCoefficients(space);
    check_av(
        sws_setColorspaceDetails(converter, coefficients, frame->color_range == AVCOL_RANGE_JPEG,
                                 coefficients, 1, 0, 1 << 16, 1 << 16),
        "unsupported video color conversion");
    const int pitch = (width * 3 + 31) / 32 * 32;
    std::vector<uint8_t> padded(size_t(pitch) * height + AV_INPUT_BUFFER_PADDING_SIZE);
    uint8_t* dst[] = {padded.data(), nullptr, nullptr, nullptr};
    int stride[] = {pitch, 0, 0, 0};
    if (sws_scale(converter, frame->data, frame->linesize, 0, height, dst, stride) != height)
      throw std::invalid_argument("cannot convert video frame to RGB");
    std::vector<uint8_t> pixels(size_t(width) * height * 3);
    for (int y = 0; y < height; ++y)
      std::memcpy(pixels.data() + size_t(y) * width * 3, padded.data() + size_t(y) * pitch,
                  width * 3);
    return pixels;
  }
};
}  // namespace
#endif

VideoInput prepare_minimax_video(const minijson::Value& value, const std::string& param,
                                 size_t byte_budget) {
#ifndef DGPP_ENABLE_VIDEO
  (void)value;
  (void)byte_budget;
  throw ImageInputError("video support requires a build with DGPP_ENABLE_VIDEO", param);
#else
  const auto* url = value.find("url");
  if (!value.is_object() || !url || !url->is_string())
    throw ImageInputError("video_url must contain a string url", param + ".url");
  double fps = 1;
  if (const auto* rate = value.find("fps")) {
    if (!rate->is_number() || !std::isfinite(rate->as_double()) || rate->as_double() <= 0 ||
        rate->as_double() > 30)
      throw ImageInputError("video fps must be greater than zero and at most 30", param + ".fps");
    fps = rate->as_double();
  }
  const auto uri = url->as_string();
  const bool mp4 = uri.starts_with("data:video/mp4;base64,");
  const bool webm = uri.starts_with("data:video/webm;base64,");
  if (!mp4 && !webm)
    throw ImageInputError("use a base64 data URI with video/mp4 or video/webm", param + ".url");
  try {
    const auto bytes = decode_base64(uri.substr(uri.find(',') + 1), 32ull << 20);
    if ((mp4 && (bytes.size() < 12 || bytes.substr(4, 4) != "ftyp")) ||
        (webm && !std::string_view(bytes).starts_with("\x1a\x45\xdf\xa3")))
      throw std::invalid_argument("video MIME type does not match its bytes");
    std::vector<double> times;
    double source_fps;
    {
      Decoder input(bytes);
      source_fps = input.fps;
      input.decode([&](int, double time) { times.push_back(time); });
    }
    const double duration = times.back() + 1 / source_fps;
    const int count =
        std::min(int(times.size()), int(std::clamp(std::nearbyint(duration * fps), 4.0, 768.0)));
    std::vector<int> indices;
    for (int i = 0; i < count; ++i)
      indices.push_back(
          count == 1 ? 0 : int(std::nearbyint(double(i) * (times.size() - 1) / (count - 1))));
    // The processor's total pixel budget applies to raw selected frames.
    // Reserve an extra frame before sizing an odd final temporal pair.
    const int padded = (count + 1) / 2 * 2;
    const int max_tokens = std::min(768, 57600 / padded);
    const ImagePreprocess preprocess{28, 4, max_tokens, false, max_tokens};
    Decoder input(bytes);
    std::vector<ImageInput> frames;
    size_t next = 0;
    input.decode([&](int index, double) {
      if (next == indices.size() || index != indices[next]) return;
      auto rgb = input.rgb();
      auto image = resize_image(rgb.data(), input.width, input.height, preprocess);
      if (image.rgb.size() > byte_budget / padded)
        throw std::invalid_argument("decoded video exceeds the request byte limit");
      frames.push_back(std::move(image));
      ++next;
    });
    if (frames.size() != indices.size())
      throw std::invalid_argument("video frame count changed during decode");
    if (count % 2) frames.push_back(frames.back());
    VideoInput out;
    for (int at = 0; at < padded; at += 8) {
      ImageInput segment = std::move(frames[at]);
      segment.frames = std::min(8, padded - at);
      segment.tokens *= segment.frames / 2;
      for (int i = 1; i < segment.frames; ++i)
        segment.rgb.insert(segment.rgb.end(), frames[at + i].rgb.begin(), frames[at + i].rgb.end());
      validate_image_pixels(segment, true);
      for (int i = 0; i < segment.frames; i += 2) out.timestamps.push_back(times[indices[at + i]]);
      out.segments.push_back(std::move(segment));
    }
    return out;
  } catch (const std::invalid_argument& e) {
    throw ImageInputError(e.what(), param + ".url");
  }
#endif
}
}  // namespace dgpp::serve
