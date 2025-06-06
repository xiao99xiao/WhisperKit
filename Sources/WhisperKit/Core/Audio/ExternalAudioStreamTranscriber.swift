//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import Foundation

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension ExternalAudioStreamTranscriber {
  public struct State {
    public var isProcessing: Bool = false
    public var currentFallbacks: Int = 0
    public var lastBufferSize: Int = 0
    public var lastConfirmedSegmentEndSeconds: Float = 0
    public var bufferEnergy: [Float] = []
    public var currentText: String = ""
    public var confirmedSegments: [TranscriptionSegment] = []
    public var unconfirmedSegments: [TranscriptionSegment] = []
    public var unconfirmedText: [String] = []
  }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public typealias ExternalAudioStreamTranscriberCallback = (
  ExternalAudioStreamTranscriber.State, ExternalAudioStreamTranscriber.State
) -> Void

/// Responsible for processing external audio samples and transcribing them in real-time.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public actor ExternalAudioStreamTranscriber {
  private var state: ExternalAudioStreamTranscriber.State = .init() {
    didSet {
      stateChangeCallback?(oldValue, state)
    }
  }

  private let stateChangeCallback: ExternalAudioStreamTranscriberCallback?
  private let requiredSegmentsForConfirmation: Int
  private let useVAD: Bool
  private let silenceThreshold: Float
  private let compressionCheckWindow: Int
  private let transcribeTask: TranscribeTask
  private let decodingOptions: DecodingOptions

  // 存储外部传入的音频样本
  private var audioSamples: [Float] = []
  private var audioEnergy: [(rel: Float, avg: Float, max: Float, min: Float)] = []
  private var relativeEnergyWindow: Int = 20

  public init(
    audioEncoder: any AudioEncoding,
    featureExtractor: any FeatureExtracting,
    segmentSeeker: any SegmentSeeking,
    textDecoder: any TextDecoding,
    tokenizer: any WhisperTokenizer,
    decodingOptions: DecodingOptions,
    requiredSegmentsForConfirmation: Int = 2,
    silenceThreshold: Float = 0.3,
    compressionCheckWindow: Int = 60,
    useVAD: Bool = true,
    stateChangeCallback: ExternalAudioStreamTranscriberCallback?
  ) {
    self.transcribeTask = TranscribeTask(
      currentTimings: TranscriptionTimings(),
      progress: Progress(),
      audioEncoder: audioEncoder,
      featureExtractor: featureExtractor,
      segmentSeeker: segmentSeeker,
      textDecoder: textDecoder,
      tokenizer: tokenizer
    )
    self.decodingOptions = decodingOptions
    self.requiredSegmentsForConfirmation = requiredSegmentsForConfirmation
    self.silenceThreshold = silenceThreshold
    self.compressionCheckWindow = compressionCheckWindow
    self.useVAD = useVAD
    self.stateChangeCallback = stateChangeCallback
  }

  /// 添加新的音频样本到处理队列
  public func appendAudioSamples(_ samples: [Float]) {
    audioSamples.append(contentsOf: samples)
  }

  /// 开始处理音频流
  public func startStreamTranscription() async throws {
    guard !state.isProcessing else { return }
    state.isProcessing = true
    Task { [weak self] in
      await self?.realtimeLoop()
    }
    Logging.info("External audio stream transcription has started")
  }

  /// 停止处理音频流
  public func stopStreamTranscription() {
    state.isProcessing = false
    Logging.info("External audio stream transcription has ended")
  }

  /// 清空音频缓冲区
  public func clearAudioBuffer() {
    audioSamples.removeAll()
    state.lastBufferSize = 0
  }

  private func realtimeLoop() async {
    while state.isProcessing {
      do {
        try await transcribeCurrentBuffer()
      } catch {
        Logging.error("Error: \(error.localizedDescription)")
        break
      }
    }
  }

  private func onProgressCallback(_ progress: TranscriptionProgress) {
    let fallbacks = Int(progress.timings.totalDecodingFallbacks)
    if progress.text.count < state.currentText.count {
      if fallbacks == state.currentFallbacks {
        state.unconfirmedText.append(state.currentText)
      } else {
        Logging.info("Fallback occured: \(fallbacks)")
      }
    }
    state.currentText = progress.text
    state.currentFallbacks = fallbacks
  }

  private func transcribeCurrentBuffer() async throws {
    // 计算下一个缓冲区的大小和持续时间
    let nextBufferSize = audioSamples.count - state.lastBufferSize
    let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

    // 只有当缓冲区至少有1秒的音频时才进行转录
    guard nextBufferSize > WhisperKit.sampleRate else {  // 确保至少有1秒的音频
      if state.currentText == "" {
        state.currentText = "Waiting for audio..."
      }
      return try await Task.sleep(nanoseconds: 1_000_000_000)  // 休眠1秒等待下一个缓冲区
    }

    if useVAD {
      let voiceDetected = AudioProcessor.isVoiceDetected(
        in: audioEnergy.map { $0.rel },
        nextBufferInSeconds: nextBufferSeconds,
        silenceThreshold: silenceThreshold
      )
      // 只有当检测到声音时才进行转录
      if !voiceDetected {
        Logging.debug("No voice detected, skipping transcribe")
        if state.currentText == "" {
          state.currentText = "Waiting for speech..."
        }
        return try await Task.sleep(nanoseconds: 1_000_000_000)  // 休眠1秒
      }
    }

    // 运行转录
    state.lastBufferSize = audioSamples.count

    let transcription = try await transcribeAudioSamples(Array(audioSamples))

    state.currentText = ""
    state.unconfirmedText = []
    let segments = transcription.segments

    // 处理确认和未确认的文本段
    if segments.count > requiredSegmentsForConfirmation {
      let numberOfSegmentsToConfirm = segments.count - requiredSegmentsForConfirmation
      let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
      let remainingSegments = Array(segments.suffix(requiredSegmentsForConfirmation))

      if let lastConfirmedSegment = confirmedSegmentsArray.last,
        lastConfirmedSegment.end > state.lastConfirmedSegmentEndSeconds
      {
        state.lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end

        if !state.confirmedSegments.contains(confirmedSegmentsArray) {
          state.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
        }
      }

      state.unconfirmedSegments = remainingSegments
    } else {
      state.unconfirmedSegments = segments
    }
  }

  private func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult {
    var options = decodingOptions
    options.clipTimestamps = [state.lastConfirmedSegmentEndSeconds]
    let checkWindow = compressionCheckWindow
    return try await transcribeTask.run(audioArray: samples, decodeOptions: options) {
      [weak self] progress in
      Task { [weak self] in
        await self?.onProgressCallback(progress)
      }
      return ExternalAudioStreamTranscriber.shouldStopEarly(
        progress: progress, options: options, compressionCheckWindow: checkWindow)
    }
  }

  private static func shouldStopEarly(
    progress: TranscriptionProgress,
    options: DecodingOptions,
    compressionCheckWindow: Int
  ) -> Bool? {
    let currentTokens = progress.tokens
    if currentTokens.count > compressionCheckWindow {
      let checkTokens: [Int] = currentTokens.suffix(compressionCheckWindow)
      let compressionRatio = compressionRatio(of: checkTokens)
      if compressionRatio > options.compressionRatioThreshold ?? 0.0 {
        return false
      }
    }
    if let avgLogprob = progress.avgLogprob, let logProbThreshold = options.logProbThreshold {
      if avgLogprob < logProbThreshold {
        return false
      }
    }
    return nil
  }

  // 计算音频能量
  private func calculateAudioEnergy(_ samples: ArraySlice<Float>) -> Float {
    let samplesArray = Array(samples)
    var rmsEnergy: Float = 0.0
    var minEnergy: Float = 0.0
    var maxEnergy: Float = 0.0

    // Calculate the root mean square of the signal
    vDSP_rmsqv(samplesArray, 1, &rmsEnergy, vDSP_Length(samplesArray.count))

    // Calculate the maximum sample value of the signal
    vDSP_maxmgv(samplesArray, 1, &maxEnergy, vDSP_Length(samplesArray.count))

    // Calculate the minimum sample value of the signal
    vDSP_minmgv(samplesArray, 1, &minEnergy, vDSP_Length(samplesArray.count))

    // Find the lowest average energy of the last 20 buffers ~2 seconds
    let minAvgEnergy = self.audioEnergy.suffix(relativeEnergyWindow).reduce(Float.infinity) {
      min($0, $1.avg)
    }
    let relativeEnergy = AudioProcessor.calculateRelativeEnergy(
      of: samplesArray, relativeTo: minAvgEnergy)

    // Update energy for buffers with valid data
    let newEnergy = (relativeEnergy, rmsEnergy, maxEnergy, minEnergy)
    self.audioEnergy.append(newEnergy)

    return relativeEnergy
  }

  private func isVoiceDetected(_ samples: ArraySlice<Float>) -> Bool {
    let relativeEnergy = calculateAudioEnergy(samples)
    return relativeEnergy > silenceThreshold
  }
}
