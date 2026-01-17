#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// Geographic distance using Haversine
double getDistance(double lat1, double lon1, double lat2, double lon2) {
    double dLat = (lat2 - lat1) * 3.14159265358979323846 / 180.0;
    double dLon = (lon2 - lon1) * 3.14159265358979323846 / 180.0;
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(lat1 * 3.14159265358979323846 / 180.0) * cos(lat2 * 3.14159265358979323846 / 180.0) *
               sin(dLon / 2) * sin(dLon / 2);
    return 2 * atan2(sqrt(a), sqrt(1 - a)) * 6371000.0; 
}

std::string clean(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '\"'), s.end());
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    return s;
}

bool isValidFilename(std::string filename) {
    std::transform(filename.begin(), filename.end(), filename.begin(), ::toupper);
    return (filename.find("TOTALITY") != std::string::npos) && 
           (filename.find("10") != std::string::npos);
}

void findGlobalMatch(double tLat, double tLon) {
    double minLineDist = 1e18;
    std::vector<std::string> bestRow;
    std::string bestFile = "";

    // Move up one directory level
    fs::path searchRoot = fs::current_path().parent_path();
    
    std::cout << "Base Directory: " << searchRoot << std::endl;
    std::cout << "Scanning for CSVs (TOTALITY +/- 10min)..." << std::endl;

    // Search starting from parent directory
    for (const auto& entry : fs::recursive_directory_iterator(searchRoot)) {
        // Skip hidden folders
        if (entry.is_directory() && entry.path().filename().string()[0] == '.') {
            continue; 
        }

        std::string filename = entry.path().filename().string();
        
        if (entry.is_regular_file() && 
            entry.path().extension() == ".csv" && 
            isValidFilename(filename)) {
            
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            std::string line, header;
            std::getline(file, header); 

            while (std::getline(file, line)) {
                if (line.empty()) continue;
                std::stringstream ss(line);
                std::string cell;
                std::vector<std::string> row;

                while (std::getline(ss, cell, ',')) {
                    row.push_back(clean(cell));
                }

                if (row.size() < 6) continue;

                try {
                    double pLat = std::stod(row[4]);
                    double pLon = std::stod(row[5]);

                    double currentDist = getDistance(tLat, tLon, pLat, pLon);
                    if (currentDist < minLineDist) {
                        minLineDist = currentDist;
                        bestRow = row;
                        bestFile = entry.path().string();
                    }
                } catch (...) { continue; }
            }
        }
    }

    if (!bestRow.empty()) {
        std::cout << "\n--- GLOBAL BEST MATCH FOUND ---" << std::endl;
        std::cout << "File: " << bestFile << std::endl;
        std::cout << "Time: " << bestRow[0] << " | Distance: " << std::fixed << std::setprecision(2) << minLineDist << "m" << std::endl;

        std::ofstream out("output.json");
        out << std::fixed << std::setprecision(6);
        out << "{\n  \"type\": \"FeatureCollection\",\n  \"features\": [\n";
        out << "    { \"type\": \"Feature\", \"properties\": { \"title\": \"\", \"marker-symbol\": \"https://www.freeiconspng.com/uploads/camera-icon-21.png\" }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [" << bestRow[5] << "," << bestRow[4] << ",0,0] } },\n";
        out << "    { \"type\": \"Feature\", \"properties\": { \"title\": \"\", \"marker-symbol\": \"hiking-uphill\", \"marker-color\": \"000000\" }, \"geometry\": { \"type\": \"Point\", \"coordinates\": [" << bestRow[3] << "," << bestRow[2] << ",0,0] } },\n";
        out << "    { \"type\": \"Feature\", \"properties\": { \"title\": \" ≈ " << bestRow[0] << "\", \"stroke\": \"#FF0000\", \"stroke-width\": 2 }, \"geometry\": { \"type\": \"LineString\", \"coordinates\": [[" << bestRow[5] << "," << bestRow[4] << "],[" << bestRow[3] << "," << bestRow[2] << "]] } }\n";
        out << "  ]\n}";
        out.close();
        std::cout << "JSON saved to current directory: output.json" << std::endl;
    } else {
        std::cout << "No matching files found." << std::endl;
    }
}

int main() {
    std::string input;
    while (true) {
        std::cout << "\nEnter Lat Long (or 'q' to quit): ";
        if (!(std::cin >> input) || input == "q") break;
        
        try {
            double lat = std::stod(input);
            if (std::cin.peek() == ',') std::cin.ignore();
            double lon;
            if (!(std::cin >> lon)) break;
            
            findGlobalMatch(lat, lon);
        } catch (...) { 
            std::cout << "Format error. Try: 38.8, -106.9" << std::endl;
            std::cin.clear();
            std::cin.ignore(1000, '\n');
        }
    }
    return 0;
}