#include "stdafx.h"
#include "file_tester.h"
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <utility>
#include <fstream>

std::pair<std::vector<std::string>, std::vector<std::string>> file_tester::obtainFileNames() {
	std::vector<std::string> directories;
	std::vector<std::string> fileNames;

	for (const auto& entry : std::filesystem::recursive_directory_iterator(INPUT_TEST_DIR)) {
		if (entry.is_directory()) {
			directories.push_back(entry.path().string());
			continue;
		}
		fileNames.push_back(entry.path().string());
	}
	return std::make_pair(directories, fileNames);
}

std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
}

void file_tester::createInfoFile(int testNumber, int iterations, int Kclusters, int patchSize, std::string heuristicFuncName) {
	// create properties file
	std::ofstream outfile(std::string(OUTPUT_TEST_DIR) + std::to_string(testNumber) + std::string("\\") + TEST_PROPS_FILENAME);
	outfile << std::string("------------------------ TEST #") + std::to_string(testNumber) + std::string(" INFO ------------------------") << "\n\n";
	outfile << "ITERATIONS: " << iterations << "\n";
	outfile << "PATCH SIZE: " << patchSize << "\n";
	outfile << "NUMBER OF CLUSTERS: " << Kclusters << "\n";
	outfile << "HEURISTIC FUNCTION USED: " << heuristicFuncName << "\n";
}

void file_tester::makeTestDirs(int testNumber, std::vector<std::string> dirs, int iterations, int Kclusters, int patchSize, std::string heuristicFuncName) {
	for (const auto& dir : dirs) {
		std::string dirName = std::string(OUTPUT_TEST_DIR) + std::to_string(testNumber) + dir.substr(INPUT_TEST_DIR_LEN);
		dirName = replaceAll(dirName, std::string("\\"), std::string("\\\\"));
		if (std::filesystem::remove_all(dirName.c_str())) {
			std::cout << "Test #" << std::to_string(testNumber) << ": Deleted dir: " << dirName << "\n";
		}
		else {
			std::cout << "Test #" << std::to_string(testNumber) << ": Dir not deleted or it doesnt exist: " << dirName << "\n";
		}

		if (std::filesystem::create_directories(dirName.c_str())) {
			std::cout << "Test #" << std::to_string(testNumber) << ": Created dir: " << dirName << "\n";
		}
		else {
			std::cout << "Test #" << std::to_string(testNumber) << ": Could not create dir: " << dirName << "\n";
		}
	}
	file_tester::createInfoFile(testNumber, iterations, Kclusters, patchSize, heuristicFuncName);
}