#include "PositionBasedDynamics.h"
#include "PositionBasedElasticRod.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "global_param.h"

using namespace PBD;

// Function to write particle positions to CSV file
void writeParticlePositionsToCSV(const std::string& filename,
                                  const std::vector<std::vector<Eigen::Vector3f>>& allPositions,
                                  int numParticles) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header
    file << "Step";
    for (int i = 0; i < numParticles; i++) {
        file << ",Particle" << i << "_X,Particle" << i << "_Y,Particle" << i << "_Z";
    }
    file << std::endl;

    // Write data
    for (int step = 0; step < allPositions.size(); step++) {
        file << step;
        for (int i = 0; i < numParticles; i++) {
            file << "," << std::fixed << std::setprecision(6)
                 << allPositions[step][i].x() << ","
                 << allPositions[step][i].y() << ","
                 << allPositions[step][i].z();
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Particle positions saved to " << filename << std::endl;
}

// Function to write performance metrics to CSV file
void writePerformanceMetricsToCSV(const std::string& filename,
                                   const std::vector<std::string>& solverNames,
                                   const std::vector<double>& avgTimes,
                                   const std::vector<double>& totalTimes,
                                   const std::vector<int>& numSteps) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header
    file << "Solver,Average_Time_Per_Step(ms),Total_Time(ms),Num_Steps,FPS" << std::endl;

    // Write data
    for (size_t i = 0; i < solverNames.size(); i++) {
        double fps = (numSteps[i] > 0) ? (1000.0 / avgTimes[i]) : 0.0;
        file << solverNames[i] << ","
             << std::fixed << std::setprecision(3) << avgTimes[i] << ","
             << totalTimes[i] << ","
             << numSteps[i] << ","
             << fps << std::endl;
    }

    file.close();
    std::cout << "Performance metrics saved to " << filename << std::endl;
}

// Test function for basic distance constraint
void testDistanceConstraint() {
    std::cout << "\n=== Testing Distance Constraint ===" << std::endl;

    // Create two particles
    Eigen::Vector3f p0(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f p1(2.0f, 0.0f, 0.0f);

    float invMass0 = 1.0f;
    float invMass1 = 1.0f;
    float restLength = 1.0f;
    float compressionStiffness = 0.5f;
    float stretchStiffness = 0.5f;

    Eigen::Vector3f corr0, corr1;

    std::cout << "Initial positions:" << std::endl;
    std::cout << "P0: (" << p0.x() << ", " << p0.y() << ", " << p0.z() << ")" << std::endl;
    std::cout << "P1: (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ")" << std::endl;
    std::cout << "Rest length: " << restLength << std::endl;

    bool success = PositionBasedDynamics::solve_DistanceConstraint(
        p0, invMass0, p1, invMass1, restLength, compressionStiffness, stretchStiffness, corr0, corr1);

    if (success) {
        std::cout << "Corrections:" << std::endl;
        std::cout << "Corr0: (" << corr0.x() << ", " << corr0.y() << ", " << corr0.z() << ")" << std::endl;
        std::cout << "Corr1: (" << corr1.x() << ", " << corr1.y() << ", " << corr1.z() << ")" << std::endl;

        // Apply corrections
        Eigen::Vector3f newP0 = p0 + corr0;
        Eigen::Vector3f newP1 = p1 + corr1;

        std::cout << "New positions:" << std::endl;
        std::cout << "P0: (" << newP0.x() << ", " << newP0.y() << ", " << newP0.z() << ")" << std::endl;
        std::cout << "P1: (" << newP1.x() << ", " << newP1.y() << ", " << newP1.z() << ")" << std::endl;

        float newDistance = (newP1 - newP0).norm();
        std::cout << "New distance: " << newDistance << std::endl;
    }
}

// Test function for dihedral constraint (bending)
void testDihedralConstraint() {
    std::cout << "\n=== Testing Dihedral Constraint ===" << std::endl;

    // Create four particles forming two adjacent triangles
    Eigen::Vector3f p0(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f p1(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f p2(0.5f, 1.0f, 0.0f);
    Eigen::Vector3f p3(1.5f, 1.0f, 0.0f);

    float invMass0 = 1.0f, invMass1 = 1.0f, invMass2 = 1.0f, invMass3 = 1.0f;
    float restAngle = M_PI / 4.0f; // 45 degrees
    float stiffness = 0.5f;

    Eigen::Vector3f corr0, corr1, corr2, corr3;

    std::cout << "Initial positions:" << std::endl;
    std::cout << "P0: (" << p0.x() << ", " << p0.y() << ", " << p0.z() << ")" << std::endl;
    std::cout << "P1: (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ")" << std::endl;
    std::cout << "P2: (" << p2.x() << ", " << p2.y() << ", " << p2.z() << ")" << std::endl;
    std::cout << "P3: (" << p3.x() << ", " << p3.y() << ", " << p3.z() << ")" << std::endl;
    std::cout << "Rest angle: " << restAngle * 180.0f / M_PI << " degrees" << std::endl;

    bool success = PositionBasedDynamics::solve_DihedralConstraint(
        p0, invMass0, p1, invMass1, p2, invMass2, p3, invMass3, restAngle, stiffness,
        corr0, corr1, corr2, corr3);

    if (success) {
        std::cout << "Corrections:" << std::endl;
        std::cout << "Corr0: (" << corr0.x() << ", " << corr0.y() << ", " << corr0.z() << ")" << std::endl;
        std::cout << "Corr1: (" << corr1.x() << ", " << corr1.y() << ", " << corr1.z() << ")" << std::endl;
        std::cout << "Corr2: (" << corr2.x() << ", " << corr2.y() << ", " << corr2.z() << ")" << std::endl;
        std::cout << "Corr3: (" << corr3.x() << ", " << corr3.y() << ", " << corr3.z() << ")" << std::endl;
    }
}

// Test function for volume constraint
void testVolumeConstraint() {
    std::cout << "\n=== Testing Volume Constraint ===" << std::endl;

    // Create four particles forming a tetrahedron
    Eigen::Vector3f p0(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f p1(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f p2(0.5f, 1.0f, 0.0f);
    Eigen::Vector3f p3(0.5f, 0.5f, 1.0f);

    float invMass0 = 1.0f, invMass1 = 1.0f, invMass2 = 1.0f, invMass3 = 1.0f;
    float restVolume = 1.0f / 6.0f; // Volume of unit tetrahedron
    float negVolumeStiffness = 0.5f;
    float posVolumeStiffness = 0.5f;

    Eigen::Vector3f corr0, corr1, corr2, corr3;

    std::cout << "Initial positions:" << std::endl;
    std::cout << "P0: (" << p0.x() << ", " << p0.y() << ", " << p0.z() << ")" << std::endl;
    std::cout << "P1: (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ")" << std::endl;
    std::cout << "P2: (" << p2.x() << ", " << p2.y() << ", " << p2.z() << ")" << std::endl;
    std::cout << "P3: (" << p3.x() << ", " << p3.y() << ", " << p3.z() << ")" << std::endl;
    std::cout << "Rest volume: " << restVolume << std::endl;

    bool success = PositionBasedDynamics::solve_VolumeConstraint(
        p0, invMass0, p1, invMass1, p2, invMass2, p3, invMass3, restVolume,
        negVolumeStiffness, posVolumeStiffness, corr0, corr1, corr2, corr3);

    if (success) {
        std::cout << "Corrections:" << std::endl;
        std::cout << "Corr0: (" << corr0.x() << ", " << corr0.y() << ", " << corr0.z() << ")" << std::endl;
        std::cout << "Corr1: (" << corr1.x() << ", " << corr1.y() << ", " << corr1.z() << ")" << std::endl;
        std::cout << "Corr2: (" << corr2.x() << ", " << corr2.y() << ", " << corr2.z() << ")" << std::endl;
        std::cout << "Corr3: (" << corr3.x() << ", " << corr3.y() << ", " << corr3.z() << ")" << std::endl;
    }
}

// Test function for elastic rod edge constraints
void testElasticRodEdgeConstraints() {
    std::cout << "\n=== Testing Elastic Rod Edge Constraints ===" << std::endl;

    // Create three particles for elastic rod (A, B, G)
    Eigen::Vector3f pA(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f pB(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f pG(0.5f, 0.5f, 0.0f); // Ghost point

    float wA = 1.0f, wB = 1.0f, wG = 1.0f;
    float edgeKs = 0.5f;
    float edgeRestLength = 1.0f;
    float ghostEdgeRestLength = 0.5f;

    Eigen::Vector3f corrA, corrB, corrG;

    std::cout << "Initial positions:" << std::endl;
    std::cout << "PA: (" << pA.x() << ", " << pA.y() << ", " << pA.z() << ")" << std::endl;
    std::cout << "PB: (" << pB.x() << ", " << pB.y() << ", " << pB.z() << ")" << std::endl;
    std::cout << "PG: (" << pG.x() << ", " << pG.y() << ", " << pG.z() << ")" << std::endl;
    std::cout << "Edge rest length: " << edgeRestLength << std::endl;
    std::cout << "Ghost edge rest length: " << ghostEdgeRestLength << std::endl;

    bool success = PositionBasedElasticRod::ProjectEdgeConstraints(
        pA, wA, pB, wB, pG, wG, edgeKs, edgeRestLength, ghostEdgeRestLength,
        corrA, corrB, corrG);

    if (success) {
        std::cout << "Corrections:" << std::endl;
        std::cout << "CorrA: (" << corrA.x() << ", " << corrA.y() << ", " << corrA.z() << ")" << std::endl;
        std::cout << "CorrB: (" << corrB.x() << ", " << corrB.y() << ", " << corrB.z() << ")" << std::endl;
        std::cout << "CorrG: (" << corrG.x() << ", " << corrG.y() << ", " << corrG.z() << ")" << std::endl;

        // Apply corrections
        Eigen::Vector3f newPA = pA + corrA;
        Eigen::Vector3f newPB = pB + corrB;
        Eigen::Vector3f newPG = pG + corrG;

        std::cout << "New positions:" << std::endl;
        std::cout << "PA: (" << newPA.x() << ", " << newPA.y() << ", " << newPA.z() << ")" << std::endl;
        std::cout << "PB: (" << newPB.x() << ", " << newPB.y() << ", " << newPB.z() << ")" << std::endl;
        std::cout << "PG: (" << newPG.x() << ", " << newPG.y() << ", " << newPG.z() << ")" << std::endl;
    }
}

// Test function for material frame computation
void testMaterialFrame() {
    std::cout << "\n=== Testing Material Frame Computation ===" << std::endl;

    // Create three particles for material frame computation
    Eigen::Vector3f pA(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f pB(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f pG(0.5f, 0.5f, 0.0f); // Ghost point

    Eigen::Matrix3f frame;

    std::cout << "Particles:" << std::endl;
    std::cout << "PA: (" << pA.x() << ", " << pA.y() << ", " << pA.z() << ")" << std::endl;
    std::cout << "PB: (" << pB.x() << ", " << pB.y() << ", " << pB.z() << ")" << std::endl;
    std::cout << "PG: (" << pG.x() << ", " << pG.y() << ", " << pG.z() << ")" << std::endl;

    bool success = PositionBasedElasticRod::ComputeMaterialFrame(pA, pB, pG, frame);

    if (success) {
        std::cout << "Material frame:" << std::endl;
        std::cout << "Frame matrix:" << std::endl;
        std::cout << frame << std::endl;

        // Check orthogonality
        Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f orthogonality = frame * frame.transpose();
        std::cout << "Orthogonality check (should be identity):" << std::endl;
        std::cout << orthogonality << std::endl;
    }
}

// Test function for Darboux vector computation
void testDarbouxVector() {
    std::cout << "\n=== Testing Darboux Vector Computation ===" << std::endl;

    // Create two material frames
    Eigen::Matrix3f dA;
    dA << 1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f dB;
    dB << 0.7071f, -0.7071f, 0.0f,
          0.7071f,  0.7071f, 0.0f,
          0.0f,     0.0f,    1.0f;

    float mid_edge_length = 1.0f;
    Eigen::Vector3f darboux_vector;

    std::cout << "Material frame A:" << std::endl;
    std::cout << dA << std::endl;
    std::cout << "Material frame B:" << std::endl;
    std::cout << dB << std::endl;
    std::cout << "Mid edge length: " << mid_edge_length << std::endl;

    bool success = PositionBasedElasticRod::ComputeDarbouxVector(dA, dB, mid_edge_length, darboux_vector);

    if (success) {
        std::cout << "Darboux vector: (" << darboux_vector.x() << ", "
                  << darboux_vector.y() << ", " << darboux_vector.z() << ")" << std::endl;
    }
}

// Test function for shape matching constraint
void testShapeMatching() {
    std::cout << "\n=== Testing Shape Matching Constraint ===" << std::endl;

    // Create a cluster of particles
    const int numPoints = 4;
    Eigen::Vector3f x0[numPoints]; // Rest configuration
    Eigen::Vector3f x[numPoints];  // Current configuration
    float invMasses[numPoints];

    // Initialize rest configuration (square)
    x0[0] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    x0[1] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    x0[2] = Eigen::Vector3f(1.0f, 1.0f, 0.0f);
    x0[3] = Eigen::Vector3f(0.0f, 1.0f, 0.0f);

    // Initialize current configuration (deformed)
    x[0] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    x[1] = Eigen::Vector3f(1.2f, 0.1f, 0.0f);
    x[2] = Eigen::Vector3f(1.1f, 1.1f, 0.0f);
    x[3] = Eigen::Vector3f(-0.1f, 1.0f, 0.0f);

    // Initialize inverse masses
    for (int i = 0; i < numPoints; i++) {
        invMasses[i] = 1.0f;
    }

    // Initialize shape matching
    Eigen::Vector3f restCm;
    Eigen::Matrix3f invRestMat;

    bool initSuccess = PositionBasedDynamics::init_ShapeMatchingConstraint(
        x0, invMasses, numPoints, restCm, invRestMat);

    if (initSuccess) {
        std::cout << "Rest center of mass: (" << restCm.x() << ", " << restCm.y() << ", " << restCm.z() << ")" << std::endl;
        std::cout << "Inverse rest matrix:" << std::endl;
        std::cout << invRestMat << std::endl;

        // Solve shape matching constraint
        Eigen::Vector3f corr[numPoints];
        Eigen::Matrix3f rot;
        float stiffness = 0.5f;
        bool allowStretch = false;

        bool solveSuccess = PositionBasedDynamics::solve_ShapeMatchingConstraint(
            x0, x, invMasses, numPoints, restCm, invRestMat, stiffness, allowStretch, corr, &rot);

        if (solveSuccess) {
            std::cout << "Corrections:" << std::endl;
            for (int i = 0; i < numPoints; i++) {
                std::cout << "Corr[" << i << "]: (" << corr[i].x() << ", " << corr[i].y() << ", " << corr[i].z() << ")" << std::endl;
            }
            std::cout << "Rotation matrix:" << std::endl;
            std::cout << rot << std::endl;
        }
    }
}

// Test function for strain-based triangle constraint
void testStrainTriangleConstraint() {
    std::cout << "\n=== Testing Strain Triangle Constraint ===" << std::endl;

    // Create three particles forming a triangle
    Eigen::Vector3f p0(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f p1(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f p2(0.5f, 1.0f, 0.0f);

    float invMass0 = 1.0f, invMass1 = 1.0f, invMass2 = 1.0f;

    // Initialize strain triangle constraint
    Eigen::Matrix2f invRestMat;
    bool initSuccess = PositionBasedDynamics::init_StrainTriangleConstraint(p0, p1, p2, invRestMat);

    if (initSuccess) {
        std::cout << "Initial positions:" << std::endl;
        std::cout << "P0: (" << p0.x() << ", " << p0.y() << ", " << p0.z() << ")" << std::endl;
        std::cout << "P1: (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ")" << std::endl;
        std::cout << "P2: (" << p2.x() << ", " << p2.y() << ", " << p2.z() << ")" << std::endl;
        std::cout << "Inverse rest matrix:" << std::endl;
        std::cout << invRestMat << std::endl;

        // Solve strain triangle constraint
        float xxStiffness = 0.5f, yyStiffness = 0.5f, xyStiffness = 0.3f;
        bool normalizeStretch = false, normalizeShear = false;
        Eigen::Vector3f corr0, corr1, corr2;

        bool solveSuccess = PositionBasedDynamics::solve_StrainTriangleConstraint(
            p0, invMass0, p1, invMass1, p2, invMass2, invRestMat,
            xxStiffness, yyStiffness, xyStiffness, normalizeStretch, normalizeShear,
            corr0, corr1, corr2);

        if (solveSuccess) {
            std::cout << "Corrections:" << std::endl;
            std::cout << "Corr0: (" << corr0.x() << ", " << corr0.y() << ", " << corr0.z() << ")" << std::endl;
            std::cout << "Corr1: (" << corr1.x() << ", " << corr1.y() << ", " << corr1.z() << ")" << std::endl;
            std::cout << "Corr2: (" << corr2.x() << ", " << corr2.y() << ", " << corr2.z() << ")" << std::endl;
        }
    }
}

// Basic Distance Constraint Solver Simulation with Circular Motion
void runDistanceConstraintSimulation(float& length_segment) {
    std::cout << "\n=== Running Distance Constraint Solver Simulation with Circular Motion ===" << std::endl;

    const int numParticles = 10;
    std::vector<Eigen::Vector3f> positions(numParticles);
    std::vector<Eigen::Vector3f> velocities(numParticles);
    std::vector<float> invMasses(numParticles);

    // Circular motion parameters
    float time = 0.0f;
    float angularSpeed = 6.28f;
    float radius = 1.0f;
    float centerY = 0.0f;

    // Initialize particles in a line
    for (int i = 0; i < numParticles; i++) {
        positions[i] = Eigen::Vector3f(i * length_segment, centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
        velocities[i] = Eigen::Vector3f::Zero();
        invMasses[i] = 1.0f;
    }

    // Store initial positions for circular motion centers
    Eigen::Vector3f leftCenter(positions[0].x(), centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
    Eigen::Vector3f rightCenter(positions[numParticles-1].x(), centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));

    float dt = 0.016f;
    Eigen::Vector3f gravity(0.0f, -9.81f, 0.0f);
    float restLength = length_segment; //natural and undeformed length of the constraint
    float stiffness = 1.0f;

    const int numSteps = 180;
    std::vector<std::vector<Eigen::Vector3f>> allPositions;
    double totalTime = 0.0;

    for (int step = 0; step < numSteps; step++) {
        auto start = std::chrono::high_resolution_clock::now();

        allPositions.push_back(positions);

        // Apply gravity to middle particles (not endpoints)
        for (int i = 1; i < numParticles - 1; i++) {
            velocities[i] += gravity * dt;
            positions[i] += velocities[i] * dt;
        }

        // Apply circular motion to both ends
        float angle = angularSpeed * time;

        // Left endpoint circular motion
        positions[0] = Eigen::Vector3f(
            leftCenter.x() ,
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );

        // Right endpoint circular motion (opposite phase)
        positions[numParticles-1] = Eigen::Vector3f(
            rightCenter.x() ,
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );

        // Apply distance constraints
        for (int i = 0; i < numParticles - 1; i++) {
            Eigen::Vector3f corr0, corr1;
            PositionBasedDynamics::solve_DistanceConstraint(
                positions[i], invMasses[i], positions[i + 1], invMasses[i + 1],
                restLength, stiffness, stiffness, corr0, corr1);

            // Only apply corrections to middle particles, not endpoints
            if (i > 0) {
                positions[i] += corr0;
            }
            if (i < numParticles - 2) {
                positions[i + 1] += corr1;
            }
        }

        // Update velocities for middle particles
        for (int i = 1; i < numParticles - 1; i++) {
            velocities[i] = (positions[i] - allPositions[step][i]) / dt;
        }

        time += dt;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        totalTime += elapsed.count();
    }

    writeParticlePositionsToCSV("distance_constraint_simulation.csv", allPositions, numParticles);
    std::cout << "Distance Constraint Simulation completed!" << std::endl;
    std::cout << "Average time per step: " << numSteps/totalTime*1000000 << "  fps" << std::endl;
}

// Shape Matching Solver Simulation with Circular Motion
void runShapeMatchingSimulation(float& length_segment) {
    std::cout << "\n=== Running Shape Matching Solver Simulation with Circular Motion ===" << std::endl;

    const int numParticles = 9; // 3x3 grid
    std::vector<Eigen::Vector3f> positions(numParticles);
    std::vector<Eigen::Vector3f> velocities(numParticles);
    std::vector<float> invMasses(numParticles);

    // Initialize particles in a 3x3 grid
    int idx = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            positions[idx] = Eigen::Vector3f(x * length_segment, y * length_segment, 1.0f);
            velocities[idx] = Eigen::Vector3f::Zero();
            invMasses[idx] = 1.0f;
            idx++;
        }
    }

    // Circular motion parameters
    float time = 0.0f;
    float angularSpeed = 6.28f;
    float radius = 1.0f;
    float centerY = 3.0f;

    // Store initial positions for circular motion centers
    Eigen::Vector3f leftCenter(positions[0].x(), centerY, 1.0f);
    Eigen::Vector3f rightCenter(positions[2].x(), centerY, 1.0f);

    // Initialize shape matching
    Eigen::Vector3f x0[numParticles]; // Rest configuration
    for (int i = 0; i < numParticles; i++) {
        x0[i] = positions[i];
    }

    Eigen::Vector3f restCm;
    Eigen::Matrix3f invRestMat;
    PositionBasedDynamics::init_ShapeMatchingConstraint(x0, invMasses.data(), numParticles, restCm, invRestMat);

    float dt = 0.016f;
    Eigen::Vector3f gravity(0.0f, -9.81f, 0.0f);
    float stiffness = 0.8f;

    const int numSteps = 180;
    std::vector<std::vector<Eigen::Vector3f>> allPositions;
    double totalTime = 0.0;

    for (int step = 0; step < numSteps; step++) {
        auto start = std::chrono::high_resolution_clock::now();

        allPositions.push_back(positions);

        // Apply gravity to middle particles (not corner particles)
        for (int i = 0; i < numParticles; i++) {
            if (i != 0 && i != 2 && i != 6 && i != 8) { // Not corner particles
                velocities[i] += gravity * dt;
                positions[i] += velocities[i] * dt;
            }
        }

        // Apply circular motion to corner particles
        float angle = angularSpeed * time;

        // Bottom-left corner
        positions[0] = Eigen::Vector3f(
            leftCenter.x() + radius * std::cos(angle),
            leftCenter.y() + radius * std::sin(angle),
            leftCenter.z()
        );

        // Bottom-right corner (opposite phase)
        positions[2] = Eigen::Vector3f(
            rightCenter.x() + radius * std::cos(angle + M_PI),
            rightCenter.y() + radius * std::sin(angle + M_PI),
            rightCenter.z()
        );

        // Top-left corner (90 degrees offset)
        positions[6] = Eigen::Vector3f(
            leftCenter.x() + radius * std::cos(angle + M_PI/2),
            leftCenter.y() + radius * std::sin(angle + M_PI/2),
            leftCenter.z()
        );

        // Top-right corner (270 degrees offset)
        positions[8] = Eigen::Vector3f(
            rightCenter.x() + radius * std::cos(angle + 3*M_PI/2),
            rightCenter.y() + radius * std::sin(angle + 3*M_PI/2),
            rightCenter.z()
        );

        // Apply shape matching constraint
        Eigen::Vector3f corr[numParticles];
        Eigen::Matrix3f rot;
        PositionBasedDynamics::solve_ShapeMatchingConstraint(
            x0, positions.data(), invMasses.data(), numParticles, restCm, invRestMat,
            stiffness, false, corr, &rot);

        // Apply corrections only to middle particles
        for (int i = 0; i < numParticles; i++) {
            if (i != 0 && i != 2 && i != 6 && i != 8) { // Not corner particles
                positions[i] += corr[i];
            }
        }

        // Update velocities for middle particles
        for (int i = 0; i < numParticles; i++) {
            if (i != 0 && i != 2 && i != 6 && i != 8) {
                velocities[i] = (positions[i] - allPositions[step][i]) / dt;
            }
        }

        time += dt;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        totalTime += elapsed.count();
    }

    writeParticlePositionsToCSV("shape_matching_simulation.csv", allPositions, numParticles);
    std::cout << "Shape Matching Simulation completed!" << std::endl;
    std::cout << "Average time per step: " << numSteps/totalTime*1000000 << " fps" << std::endl;
}

// Strain-based Triangle Solver Simulation with Circular Motion
void runStrainTriangleSimulation(float& length_segment) {
    std::cout << "\n=== Running Strain Triangle Solver Simulation with Circular Motion ===" << std::endl;

    const int numParticles = 6; // Two triangles
    std::vector<Eigen::Vector3f> positions(numParticles);
    std::vector<Eigen::Vector3f> velocities(numParticles);
    std::vector<float> invMasses(numParticles);

    // Initialize particles forming two triangles
    positions[0] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    positions[1] = Eigen::Vector3f(length_segment, 0.0f, 1.0f);
    positions[2] = Eigen::Vector3f(length_segment/2, length_segment, 1.0f);
    positions[3] = Eigen::Vector3f(length_segment, 0.0f, 1.0f);
    positions[4] = Eigen::Vector3f(2.0f*length_segment, 0.0f, 1.0f);
    positions[5] = Eigen::Vector3f(1.5f*length_segment, 1.0f, 1.0f);

    for (int i = 0; i < numParticles; i++) {
        velocities[i] = Eigen::Vector3f::Zero();
        invMasses[i] = 1.0f;
    }

    // Circular motion parameters
    float time = 0.0f;
    float angularSpeed = 6.28f;
    float radius = 1.0f;
    float centerY = 3.0f;

    // Store initial positions for circular motion centers
    Eigen::Vector3f leftCenter(positions[0].x(), centerY, 1.0f);
    Eigen::Vector3f rightCenter(positions[4].x(), centerY, 1.0f);

    // Initialize strain triangle constraints
    Eigen::Matrix2f invRestMat1, invRestMat2;
    PositionBasedDynamics::init_StrainTriangleConstraint(positions[0], positions[1], positions[2], invRestMat1);
    PositionBasedDynamics::init_StrainTriangleConstraint(positions[3], positions[4], positions[5], invRestMat2);

    float dt = 0.016f;
    Eigen::Vector3f gravity(0.0f, -9.81f, 0.0f);
    float xxStiffness = 0.8f, yyStiffness = 0.8f, xyStiffness = 0.6f;

    const int numSteps = 180;
    std::vector<std::vector<Eigen::Vector3f>> allPositions;
    double totalTime = 0.0;

    for (int step = 0; step < numSteps; step++) {
        auto start = std::chrono::high_resolution_clock::now();

        allPositions.push_back(positions);

        // Apply gravity to middle particles (not bottom particles)
        for (int i = 2; i < numParticles; i++) { // Only top particles (2, 5)
            velocities[i] += gravity * dt;
            positions[i] += velocities[i] * dt;
        }

        // Apply circular motion to bottom particles
        float angle = angularSpeed * time;

        // Left bottom particle
        positions[0] = Eigen::Vector3f(
            leftCenter.x() + radius * std::cos(angle),
            leftCenter.y() + radius * std::sin(angle),
            leftCenter.z()
        );

        // Right bottom particle (opposite phase)
        positions[4] = Eigen::Vector3f(
            rightCenter.x() + radius * std::cos(angle + M_PI),
            rightCenter.y() + radius * std::sin(angle + M_PI),
            rightCenter.z()
        );

        // Middle bottom particles (90 and 270 degrees offset)
        positions[1] = Eigen::Vector3f(
            leftCenter.x() + radius * std::cos(angle + M_PI/2),
            leftCenter.y() + radius * std::sin(angle + M_PI/2),
            leftCenter.z()
        );

        positions[3] = Eigen::Vector3f(
            rightCenter.x() + radius * std::cos(angle + 3*M_PI/2),
            rightCenter.y() + radius * std::sin(angle + 3*M_PI/2),
            rightCenter.z()
        );

        // Apply strain triangle constraints
        Eigen::Vector3f corr0, corr1, corr2;

        // First triangle
        PositionBasedDynamics::solve_StrainTriangleConstraint(
            positions[0], invMasses[0], positions[1], invMasses[1], positions[2], invMasses[2],
            invRestMat1, xxStiffness, yyStiffness, xyStiffness, false, false, corr0, corr1, corr2);
        // Only apply corrections to top particle
        positions[2] += corr2;

        // Second triangle
        PositionBasedDynamics::solve_StrainTriangleConstraint(
            positions[3], invMasses[3], positions[4], invMasses[4], positions[5], invMasses[5],
            invRestMat2, xxStiffness, yyStiffness, xyStiffness, false, false, corr0, corr1, corr2);
        // Only apply corrections to top particle
        positions[5] += corr2;

        // Update velocities for top particles
        for (int i = 2; i < numParticles; i++) {
            velocities[i] = (positions[i] - allPositions[step][i]) / dt;
        }

        time += dt;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        totalTime += elapsed.count();
    }

    writeParticlePositionsToCSV("strain_triangle_simulation.csv", allPositions, numParticles);
    std::cout << "Strain Triangle Simulation completed!" << std::endl;
    std::cout << "Average time per step: " << numSteps/totalTime*1000000 << "  fps" << std::endl;
}

// Elastic Rod Solver Simulation with Circular Motion
void runElasticRodSimulation(float& length_segment) {
    std::cout << "\n=== Running Elastic Rod Solver Simulation with Circular Motion ===" << std::endl;

    const int num_key_nodes = 10; // N centerline particles
    const int num_ghost_points = num_key_nodes - 1; // (N-1) ghost points for interior edges only
    const int numParticles = num_key_nodes + num_ghost_points; // Total particles
    float length_ghost = 0.5*length_segment;
    std::vector<Eigen::Vector3f> positions(numParticles);
    std::vector<Eigen::Vector3f> velocities(numParticles);
    std::vector<float> invMasses(numParticles);

    // Circular motion parameters
    float time = 0.0f;
    float angularSpeed = 6.28f;
    float radius = 1.0f;
    float centerY = 0.0f;

    // Initialize centerline particles
    for (int i = 0; i < num_key_nodes; i++) {
        positions[i] = Eigen::Vector3f(i * length_segment, centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
        velocities[i] = Eigen::Vector3f::Zero();
        invMasses[i] = 1.0f;
    }

    // Initialize ghost particles using parallel transport for twist-free initialization
    // Following the paper's approach for isotropic rod with circular cross-section
    std::vector<Eigen::Vector3f> materialFrames(num_key_nodes-1); // Store material frame directions

    // Initialize first material frame (arbitrary for isotropic rod, but consistent)
    Eigen::Vector3f firstEdge = (positions[1] - positions[0]).normalized();
    Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f firstFrame = firstEdge.cross(up).normalized();
    if (firstFrame.norm() < 0.1f) {
        // If edge is nearly vertical, use a different reference
        firstFrame = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    }
    materialFrames[0] = firstFrame;

    // Parallel transport for subsequent edges to minimize twist
    for (int i = 1; i < num_key_nodes-1; i++) {
        Eigen::Vector3f prevEdge = (positions[i] - positions[i-1]).normalized();
        Eigen::Vector3f currEdge = (positions[i+1] - positions[i]).normalized();

        // Parallel transport: rotate the material frame to maintain minimal twist
        Eigen::Vector3f rotationAxis = prevEdge.cross(currEdge);
        float rotationAngle = std::acos(std::max(-1.0f, std::min(1.0f, prevEdge.dot(currEdge))));

        if (rotationAxis.norm() > 0.001f && rotationAngle > 0.001f) {
            rotationAxis.normalize();
            // Apply rotation using Rodrigues' formula
            Eigen::Vector3f prevFrame = materialFrames[i-1];
            Eigen::Vector3f newFrame = prevFrame * std::cos(rotationAngle) +
                                     rotationAxis.cross(prevFrame) * std::sin(rotationAngle) +
                                     rotationAxis * rotationAxis.dot(prevFrame) * (1.0f - std::cos(rotationAngle));
            materialFrames[i] = newFrame.normalized();
        } else {
            // No significant rotation, keep previous frame
            materialFrames[i] = materialFrames[i-1];
        }
    }

    // Initialize ghost particles at interior edge midpoints with material frame alignment
    // Ghost points are only for interior edges (indices 1 to N-2), not endpoints
    for (int i = 0; i < num_ghost_points; i++) {
        int ghostIndex = num_key_nodes + i;
        int edgeIndex = i; // Edge i connects particles i and i+1

        // Calculate midpoint of edge i
        Eigen::Vector3f pA = positions[edgeIndex];
        Eigen::Vector3f pB = positions[edgeIndex + 1];
        Eigen::Vector3f midpoint = 0.5f * (pA + pB);

        // Position ghost particle using material frame direction
        positions[ghostIndex] = midpoint + length_ghost * materialFrames[edgeIndex];
        velocities[ghostIndex] = Eigen::Vector3f::Zero();
        invMasses[ghostIndex] = 1.0f;
    }

    // Store initial positions for circular motion centers
    Eigen::Vector3f leftCenter(positions[0].x(), centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
    Eigen::Vector3f rightCenter(positions[num_key_nodes-1].x(), centerY + radius * std::sin(0.0), 1.0f + radius * std::cos(0.0));

    float dt = 0.016f;
    Eigen::Vector3f gravity(0.0f, -9.81f, 0.0f);
    float restLength = length_segment;
    
    float edgeKs = 1.0f;//Stiffness matrix
    float edgeRestLength = length_segment;
    float ghostEdgeRestLength = length_ghost;

    const int numSteps = 180;
    std::vector<std::vector<Eigen::Vector3f>> allPositions;
    double totalTime = 0.0;

    for (int step = 0; step < numSteps; step++) {
        auto start = std::chrono::high_resolution_clock::now();

        allPositions.push_back(positions);

        // Store previous velocities for gravity redistribution calculation
        std::vector<Eigen::Vector3f> prevVelocities = velocities;

        /*** Update middle points and apply circular motion to both ends ***/
        // Apply gravity to middle centerline particles
        for (int i = 1; i < num_key_nodes-1; i++) {//exclude both ends
            velocities[i] += gravity * dt;
            positions[i] += velocities[i] * dt;
        }

        // Apply circular motion to endpoints
        float angle = angularSpeed * time;

        // Left endpoint (centerline)
        positions[0] = Eigen::Vector3f(
            leftCenter.x() ,
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );

        // Right endpoint (centerline)
        positions[num_key_nodes-1] = Eigen::Vector3f(
            rightCenter.x(),
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );
        /*** End of Update middle points and apply circular motion to both ends ***/

        /*** Update ghost points with gravity redistribution ***/
        // Apply gravity redistribution to ghost particles (following paper's heuristic)
        for (int i = 0; i < num_ghost_points; i++) {
            int ghostIndex = num_key_nodes + i;
            int edgeIndex = i; // Edge i connects particles i and i+1

            // Calculate midpoint velocity (average of endpoint velocities)
            Eigen::Vector3f midVelocity = 0.5f * (velocities[edgeIndex] + velocities[edgeIndex + 1]);

            // Calculate acceleration at midpoint by forward differentiation
            Eigen::Vector3f prevMidVelocity = 0.5f * (prevVelocities[edgeIndex] + prevVelocities[edgeIndex + 1]);
            Eigen::Vector3f midAcceleration = (midVelocity - prevMidVelocity) / dt;

            // Compute gravity redistribution ratio
            float gravityMagnitude = gravity.norm();
            float ratio = 0.0f;
            if (gravityMagnitude > 0.001f) {
                ratio = (midAcceleration.dot(gravity)) / (gravityMagnitude * gravityMagnitude);
            }

            // Apply redistributed gravity force to ghost point
            Eigen::Vector3f redistributedGravity = ratio * gravity;
            velocities[ghostIndex] += redistributedGravity * dt;
            positions[ghostIndex] += velocities[ghostIndex] * dt;
        }
        /*** End of Update ghost points with gravity redistribution ***/

        //Consider constraints for edge-to-edge and edge-to-ghost constraints.
        // Apply elastic rod edge constraints for all edges
        for (int i = 0; i < num_key_nodes-1; i++) {
            Eigen::Vector3f corrA, corrB, corrG;

            // For interior edges (i > 0 and i < num_key_nodes-2), we have ghost points
            if (i < num_ghost_points) {
                PositionBasedElasticRod::ProjectEdgeConstraints(
                    positions[i], invMasses[i], positions[i+1], invMasses[i+1],
                    positions[num_key_nodes + i], invMasses[num_key_nodes + i], edgeKs, edgeRestLength, ghostEdgeRestLength,
                    corrA, corrB, corrG);

                // Only apply corrections to middle particles, not endpoints
                // corrA & corrB: Keep the centerline particles at the correct distance and maintain the rod's shape
                // corrG: Keep the ghost particle at the correct distance from the centerline,
                //which helps maintain the rod's material frame and prevents twisting
                if (i > 0) {
                    positions[i] += corrA;
                }
                if (i < num_key_nodes-2) {
                    positions[i+1] += corrB;
                }
                // Apply ghost corrections to all ghost particles
                positions[num_key_nodes + i] += corrG;
            } else {
                // For the last edge (endpoint constraint), we only have centerline particles
                // This implements the endpoint constraint |q-p0| = 0
                // For now, we'll use a simple distance constraint between the last two particles
                Eigen::Vector3f delta = positions[i+1] - positions[i];
                float dist = delta.norm();
                if (dist > 0.001f) {
                    float correction = (dist - edgeRestLength) / dist;
                    Eigen::Vector3f correctionVector = delta * correction * 0.5f;

                    // Only apply to the second-to-last particle (not the endpoint)
                    if (i < num_key_nodes-2) {
                        positions[i] += correctionVector;
                    }
                }
            }
        }
        // End of edge constraints.

        //Update the velocity based on the optimized position..
        // Update velocities for middle particles (except endpoints)
        for (int i = 1; i < num_key_nodes-1; i++) {//remove both ends.
            velocities[i] = (positions[i] - allPositions[step][i]) / dt;
        }

        // Update velocities for all ghost particles
        for (int i = 0; i < num_ghost_points; i++) {
            int ghostIndex = num_key_nodes + i;
            velocities[ghostIndex] = (positions[ghostIndex] - allPositions[step][ghostIndex]) / dt;
        }

        time += dt;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        totalTime += elapsed.count();
    }

    writeParticlePositionsToCSV("elastic_rod_simulation.csv", allPositions, numParticles);
    std::cout << "Elastic Rod Simulation completed!" << std::endl;
    std::cout << "Average time per step: " << numSteps/totalTime*1000000 << "  fps" << std::endl;
}

// Circular motion simulation (original)
void runCircularMotionSimulation(float& length_segment) {
    std::cout << "\n=== Running Circular Motion Simulation ===" << std::endl;

    const int numParticles = 10;
    std::vector<Eigen::Vector3f> positions(numParticles);
    std::vector<Eigen::Vector3f> velocities(numParticles);
    std::vector<float> invMasses(numParticles);

    // Circular motion parameters
    float time = 0.0f;
    float angularSpeed = 6.28f;
    float radius = 1.0f;
    float centerY = 0.0f;

    // Initialize particles in a line
    for (int i = 0; i < numParticles; i++) {
        positions[i] = Eigen::Vector3f(i * length_segment, centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
        velocities[i] = Eigen::Vector3f::Zero();
        invMasses[i] = 1.0f;
    }

    float dt = 0.016f;
    Eigen::Vector3f gravity(0.0f, -9.81f, 0.0f);
    float restLength = length_segment;
    float stiffness = 1.0f;

    Eigen::Vector3f leftCenter(positions[0].x(), centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));
    Eigen::Vector3f rightCenter(positions[numParticles-1].x(), centerY+ radius * std::sin(0.0), 1.0f+radius*std::cos(0.0));

    const int numSteps = 180;
    std::vector<std::vector<Eigen::Vector3f>> allPositions;
    double totalTime = 0.0;

    for (int step = 0; step < numSteps; step++) {
        auto start = std::chrono::high_resolution_clock::now();

        allPositions.push_back(positions);

        // Apply gravity to middle particles
        for (int i = 1; i < numParticles - 1; i++) {
            velocities[i] += gravity * dt;
            positions[i] += velocities[i] * dt;
        }

        // Apply circular motion to both ends
        float angle = angularSpeed * time;
        positions[0] = Eigen::Vector3f(
            leftCenter.x() ,
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );
        positions[numParticles-1] = Eigen::Vector3f(
            rightCenter.x(),
            centerY + radius * std::sin(angle),
            1.0f+ radius * std::cos(angle)
        );

        // Apply distance constraints
        for (int i = 0; i < numParticles - 1; i++) {
            Eigen::Vector3f corr0, corr1;
            PositionBasedDynamics::solve_DistanceConstraint(
                positions[i], invMasses[i], positions[i + 1], invMasses[i + 1],
                restLength, stiffness, stiffness, corr0, corr1);

            if (i > 0) positions[i] += corr0;
            if (i < numParticles - 2) positions[i + 1] += corr1;
        }

        // Update velocities for middle particles
        for (int i = 1; i < numParticles - 1; i++) {
            velocities[i] = (positions[i] - allPositions[step][i]) / dt;
        }

        time += dt;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        totalTime += elapsed.count();
    }

    writeParticlePositionsToCSV("circular_motion_simulation.csv", allPositions, numParticles);
    std::cout << "Circular Motion Simulation completed!" << std::endl;
    std::cout << "Average time per step: " << numSteps/totalTime*1000000 << "  fps" << std::endl;
}

// Comprehensive solver performance comparison
void runSolverPerformanceComparison(float& length_segment) {
    std::cout << "\n=== Running Comprehensive Solver Performance Comparison ===" << std::endl;

    std::vector<std::string> solverNames;
    std::vector<double> avgTimes;
    std::vector<double> totalTimes;
    std::vector<int> numSteps;

    // Run all simulations and collect performance data
    auto runSimulationWithTiming = [&](const std::string& name, std::function<void(float&)> simulation) {
        auto start = std::chrono::high_resolution_clock::now();
        simulation(length_segment);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total = end - start;

        solverNames.push_back(name);
        totalTimes.push_back(total.count());
        numSteps.push_back(180); // Most simulations use 150 steps
        avgTimes.push_back(total.count() / 180.0);

        std::cout << name << " - Total time: " << total.count() << "ms, Avg per step: "
                  << total.count() / 180.0 << "ms" << std::endl;
    };

    // Run each solver
    runSimulationWithTiming("Distance Constraint", runDistanceConstraintSimulation);
    //runSimulationWithTiming("Shape Matching", runShapeMatchingSimulation);
    //runSimulationWithTiming("Strain Triangle", runStrainTriangleSimulation);
    runSimulationWithTiming("Elastic Rod", runElasticRodSimulation);
    runSimulationWithTiming("Circular Motion", runCircularMotionSimulation);

    // Save performance metrics
    writePerformanceMetricsToCSV("solver_performance_comparison.csv", solverNames, avgTimes, totalTimes, numSteps);

    // Print summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Solver\t\t\tAvg Time/Step (ms)\tFPS" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for (size_t i = 0; i < solverNames.size(); i++) {
        double fps = (avgTimes[i] > 0) ? (1000.0 / avgTimes[i]) : 0.0;
        std::cout << solverNames[i] << "\t\t" << std::fixed << std::setprecision(3)
                  << avgTimes[i] << "\t\t\t" << fps << std::endl;
    }
}

int main() {
    std::cout << "Position Based Dynamics Performance Comparison Suite" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Comparing performance of various PBD solvers and generating CSV data" << std::endl;

    // Run individual constraint tests (quick validation)
    std::cout << "\n=== Running Quick Constraint Tests ===" << std::endl;
    testDistanceConstraint();
    testDihedralConstraint();
    testVolumeConstraint();
    testElasticRodEdgeConstraints();
    testMaterialFrame();
    testDarbouxVector();
    testShapeMatching();
    testStrainTriangleConstraint();

    // Run comprehensive performance comparison
    float length_segment = 0.3f;//0.3 m per segment
    runSolverPerformanceComparison(length_segment);

    std::cout << "\n=== All Tests Complete ===" << std::endl;
    std::cout << "Generated CSV files:" << std::endl;
    std::cout << "- distance_constraint_simulation.csv" << std::endl;
    std::cout << "- shape_matching_simulation.csv" << std::endl;
    std::cout << "- strain_triangle_simulation.csv" << std::endl;
    std::cout << "- elastic_rod_simulation.csv" << std::endl;
    std::cout << "- circular_motion_simulation.csv" << std::endl;
    std::cout << "- solver_performance_comparison.csv" << std::endl;
    std::cout << "\nEach CSV file contains sequential particle positions for every simulation step." << std::endl;

    return 0;
}
