import Foundation
import Security

// =====================================================================
//  AUTH MANAGER
//  Owns the signed-in doctor's session: JWT token (stored in Keychain),
//  username, and an explicit "guest" mode for anonymous analysis.
// =====================================================================

@MainActor
final class AuthManager: ObservableObject {
    @Published private(set) var token: String?
    @Published private(set) var username: String?
    @Published private(set) var isGuest: Bool = false

    private let service = "com.jaskra.app"
    private let tokenAccount = "jwt"
    private let usernameKey = "doctor.username"

    var isLoggedIn: Bool { token != nil }

    /// Whether the main app UI may be shown (logged in or explicitly a guest).
    var hasAccess: Bool { isLoggedIn || isGuest }

    init() {
        token = Keychain.read(service: service, account: tokenAccount)
        username = UserDefaults.standard.string(forKey: usernameKey)
    }

    // MARK: Actions

    func login(username: String, password: String) async throws {
        let newToken = try await GlaucomaService.shared.login(username: username, password: password)
        persist(token: newToken, username: username)
    }

    func register(username: String, password: String) async throws {
        try await GlaucomaService.shared.register(username: username, password: password)
        // Automatically sign the doctor in after a successful registration.
        try await login(username: username, password: password)
    }

    func continueAsGuest() {
        isGuest = true
    }

    /// Promote a guest into the login flow (used by "Zaloguj się" prompts).
    func exitGuest() {
        isGuest = false
    }

    func logout() {
        Keychain.delete(service: service, account: tokenAccount)
        UserDefaults.standard.removeObject(forKey: usernameKey)
        token = nil
        username = nil
        isGuest = false
    }

    // MARK: Private

    private func persist(token: String, username: String) {
        Keychain.save(token, service: service, account: tokenAccount)
        UserDefaults.standard.set(username, forKey: usernameKey)
        self.token = token
        self.username = username
        self.isGuest = false
    }
}

// =====================================================================
//  Minimal Keychain wrapper (no external dependencies)
// =====================================================================

enum Keychain {
    @discardableResult
    static func save(_ value: String, service: String, account: String) -> Bool {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(query as CFDictionary)

        var attributes = query
        attributes[kSecValueData as String] = data
        return SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess
    }

    static func read(service: String, account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else { return nil }
        return value
    }

    @discardableResult
    static func delete(service: String, account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        return SecItemDelete(query as CFDictionary) == errSecSuccess
    }
}
