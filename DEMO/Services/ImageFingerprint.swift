import UIKit
import CoreGraphics

// =====================================================================
//  IMAGE FINGERPRINT (demo only) — recognizes when a photo picked live
//  during a demo is one of our own reference fundus photos, so the app
//  can show the real pipeline result instead of a random mock.
//
//  Uses a difference hash (dHash): downsample to 9x8 grayscale, then hash
//  each row's left-to-right brightness gradient into 64 bits. Robust to
//  the resizing/recompression the Photos picker applies, since it only
//  cares about relative brightness between neighboring pixels — not
//  exact pixel values.
// =====================================================================

enum ImageFingerprint {
    /// 64-bit difference hash, or nil if the image couldn't be rendered.
    static func dHash(_ image: UIImage) -> UInt64? {
        let width = 9
        let height = 8
        guard let cgImage = image.cgImage else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var hash: UInt64 = 0
        for row in 0..<height {
            for col in 0..<(width - 1) {
                hash <<= 1
                if pixels[row * width + col] > pixels[row * width + col + 1] {
                    hash |= 1
                }
            }
        }
        return hash
    }

    static func hammingDistance(_ a: UInt64, _ b: UInt64) -> Int {
        (a ^ b).nonzeroBitCount
    }

    /// Bits (out of 64) allowed to differ and still count as "the same photo".
    /// Kept strict since fundus ROIs are all visually similar (round, bright
    /// disc roughly centered) — a loose threshold risks matching the wrong
    /// reference photo. Tune here if recognition misfires on a real device.
    static let matchThreshold = 6

    /// Finds the reference seed photo (if any) matching `image` closely
    /// enough to be considered the same source file.
    static func matchingSeed(for image: UIImage) -> DemoSeedPatient? {
        guard let targetHash = dHash(image) else { return nil }

        var best: (seed: DemoSeedPatient, distance: Int)?
        for seed in DemoPatientSeed.allForMatching {
            guard let refImage = seed.decodedRawImage, let refHash = dHash(refImage) else { continue }
            let distance = hammingDistance(targetHash, refHash)
            if best == nil || distance < best!.distance {
                best = (seed, distance)
            }
        }

        guard let best, best.distance <= matchThreshold else { return nil }
        return best.seed
    }
}

extension DemoSeedPatient {
    var decodedRawImage: UIImage? {
        Data(base64Encoded: rawImageBase64).flatMap { UIImage(data: $0) }
    }
}
