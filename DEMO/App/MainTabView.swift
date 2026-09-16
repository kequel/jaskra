import SwiftUI

// =====================================================================
//  MAIN TAB VIEW — Start · Pacjenci · Historia · Profil
// =====================================================================

struct MainTabView: View {
    var body: some View {
        TabView {
            NavigationStack {
                HomeView()
            }
            .tabItem { Label("Start", systemImage: "house.fill") }

            NavigationStack {
                PatientsListView()
            }
            .tabItem { Label("Pacjenci", systemImage: "person.2.fill") }

            NavigationStack {
                HistoryView()
            }
            .tabItem { Label("Historia", systemImage: "clock.fill") }

            NavigationStack {
                ProfileView()
            }
            .tabItem { Label("Profil", systemImage: "person.crop.circle.fill") }
        }
    }
}
