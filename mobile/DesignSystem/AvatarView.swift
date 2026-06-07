import SwiftUI

// =====================================================================
//  AVATAR VIEW
//  Circular, multi-color avatar built from an SF Symbols figure glyph.
//  Used across patient lists, history rows and profile.
// =====================================================================

struct AvatarView: View {
    let kind: AvatarKind
    let tint: AvatarTint
    var size: CGFloat = 48

    var body: some View {
        ZStack {
            Circle()
                .fill(tint.color.opacity(0.18))
            Circle()
                .strokeBorder(tint.color.opacity(0.35), lineWidth: 1)
            Image(systemName: kind.symbol)
                .font(.system(size: size * 0.5, weight: .regular))
                .foregroundStyle(tint.color)
        }
        .frame(width: size, height: size)
    }
}

/// Convenience initializer straight from a Patient.
extension AvatarView {
    init(patient: Patient, size: CGFloat = 48) {
        self.init(kind: patient.avatarKind, tint: patient.avatarTint, size: size)
    }
}

#Preview {
    HStack(spacing: 16) {
        ForEach(AvatarKind.allCases) { kind in
            AvatarView(kind: kind, tint: kind.defaultTint, size: 56)
        }
    }
    .padding()
    .background(Color.bg)
}
