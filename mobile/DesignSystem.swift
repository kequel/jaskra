import SwiftUI

// Design System

extension Color {
    static let appBackground    = Color(hex: "#080C10")
    static let cardBackground   = Color(hex: "#0E1419")
    static let cardBorder       = Color(hex: "#1C2733")
    static let accentCyan       = Color(hex: "#00D4FF")
    static let accentCyanDim    = Color(hex: "#00D4FF").opacity(0.15)
    static let positiveGreen    = Color(hex: "#00E676")
    static let warningAmber     = Color(hex: "#FFB300")
    static let dangerRed        = Color(hex: "#FF3D5A")
    static let textPrimary      = Color(hex: "#EDF2F7")
    static let textSecondary    = Color(hex: "#718096")
    static let textTertiary     = Color(hex: "#4A5568")

    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default: (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(.sRGB,
                  red: Double(r) / 255,
                  green: Double(g) / 255,
                  blue: Double(b) / 255,
                  opacity: Double(a) / 255)
    }
}

// Glowing border modifier
struct GlowBorder: ViewModifier {
    var color: Color
    var radius: CGFloat = 8

    func body(content: Content) -> some View {
        content
            .overlay(
                RoundedRectangle(cornerRadius: radius)
                    .stroke(color.opacity(0.6), lineWidth: 1)
            )
            .shadow(color: color.opacity(0.25), radius: 12, x: 0, y: 0)
    }
}

extension View {
    func glowBorder(_ color: Color, radius: CGFloat = 8) -> some View {
        modifier(GlowBorder(color: color, radius: radius))
    }
}

// Scan line animation modifier
struct ScanLineEffect: View {
    @State private var offset: CGFloat = -200

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [.clear, Color.accentCyan.opacity(0.35), .clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(height: 60)
                .offset(y: offset)
                .onAppear {
                    withAnimation(
                        .linear(duration: 2.2).repeatForever(autoreverses: false)
                    ) {
                        offset = geo.size.height + 60
                    }
                }
        }
        .clipped()
    }
}

// Pulsing circle
struct PulsingCircle: View {
    var color: Color
    @State private var scale: CGFloat = 1.0
    @State private var opacity: Double = 0.6

    var body: some View {
        Circle()
            .fill(color.opacity(opacity))
            .scaleEffect(scale)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: true)) {
                    scale = 1.18
                    opacity = 0.2
                }
            }
    }
}

// Metric Card
struct MetricCard: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.12))
                    .frame(width: 42, height: 42)
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(color)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                //.foregroundStyle(.textSecondary)
                    .textCase(.uppercase)
                    .tracking(0.8)
                Text(value)
                    .font(.system(size: 20, weight: .semibold, design: .monospaced))
                    .foregroundStyle(color)
            }
            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(Color.cardBackground)
        .glowBorder(color.opacity(0.4), radius: 14)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}
