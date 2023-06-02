/**
 * @file summaryexercise_main.cc
 * @brief NPDE homework SummaryExercise code
 * @author Samuel Russo, Jonas Bachmman
 * @date 02.06.2023
 * @copyright Developed at ETH Zurich
 */

#include <lf/assemble/assemble.h>
#include <lf/uscalfe/uscalfe.h>
#include <lf/geometry/geometry.h>
#include <lf/io/gmsh_reader.h>
#include <lf/io/vtk_writer.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh.h>
#include <lf/refinement/refinement.h>
#include <lf/fe/fe.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "summaryexercise.h"


#define Time 0.001
#define timesteps 50

// #define Time 0.03
// #define timesteps 1500
// #define Time 0.01
// #define timesteps 500
#define interval 10
#define speed 200
#define INLET_X -2.4
#define OUTLET_X 3.4
#define TOP_WALL 0.9
#define BOTTOM_WALL -0.9


// TODO 2: Complete this function.
/// @brief Volumetric averaging for translating cell centered values to
///     vertex values.
/// point_data = \sum_K |K| u_K / \sum_K |K|
/// sum over neighbour K's
/// @tparam T 
/// @param mesh_p 
/// @param cell_data 
/// @param point_data 
template<typename T>
void cell_to_vertex(
    std::shared_ptr<lf::mesh::Mesh> mesh_p,
    std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<T>> cell_data,
    std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<T>> point_data
    ) {
    lf::mesh::utils::CodimMeshDataSet<double> point_mass(mesh_p, 2, 0.);
    for(const lf::mesh::Entity* entity: mesh_p->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = entity->Geometry();
        const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
        const double area = lf::geometry::Volume(*geo_ptr);
        const T local_data = (*cell_data)(*entity);
        for (const lf::mesh::Entity *vertex : entity->SubEntities(2)) {
            // FILL ME !
            (*point_data)(*vertex) += local_data*area;
            point_mass(*vertex) += area;
        }
    }

    for(const lf::mesh::Entity *vertex: mesh_p->Entities(2)) {
        (*point_data)(*vertex) /= point_mass(*vertex);
    }
};

// TODO 4: Complete the advection step.

/// @brief 
/// @param mesh_p 
/// @param uv_cell_p 
/// @param uv_vertex_p 
/// @param normals_cell_p 
/// @param neighbour_cells_p 
/// @param tau 
/// @param vtk_writer 
void advection_step(
        std::shared_ptr<lf::mesh::Mesh> mesh_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<Eigen::Vector2d>> uv_cell_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<Eigen::Matrix<double, 2, 3>>> normals_cell_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<std::array<const lf::mesh::Entity*, 3>>> neighbour_cells_p,
        double tau,
        std::shared_ptr<lf::io::VtkWriter> vtk_writer) {
    // 1. Initialize dof handler (1 per cell)
    // TODO 4.1 fill the numbers
    const static lf::assemble::UniformFEDofHandler dofh(
        mesh_p, {{lf::base::RefEl::kPoint(), 0},
                   {lf::base::RefEl::kSegment(), 0},
                   {lf::base::RefEl::kTria(), 1},
                   {lf::base::RefEl::kQuad(), 0}}); // triangle mesh, 1 dof per cell
    const static int N_dofs = dofh.NumDofs();

    // 2. Advect uv
    Eigen::SparseMatrix<double> B(N_dofs, N_dofs);
    int bound_nnz = dofh.Mesh()->NumEntities(0) * 3;
    B.reserve(bound_nnz);

    for(const lf::mesh::Entity *cell: mesh_p->Entities(0)) {
        const int row = dofh.GlobalDofIndices(*cell)[0];
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
        const double area = lf::geometry::Volume(*geo_ptr);

        int i = 0;
        for (const lf::mesh::Entity *edge: cell->SubEntities(1)) {
            const lf::geometry::Geometry *edge_geo_ptr = edge->Geometry();
            const Eigen::MatrixXd edge_corners = lf::geometry::Corners(*edge_geo_ptr);
            const double length = lf::geometry::Volume(*edge_geo_ptr);
            const Eigen::Vector2d uv_middle = (*uv_cell_p)(*cell);
            const Eigen::Vector2d n = (*normals_cell_p)(*cell).col(i);
            const double flux = uv_middle.dot(n);

            if((*neighbour_cells_p)(*cell)[i] != nullptr) {
                if (flux >= 0) {
                    // TODO 4.2
                    B.coeffRef(row, row) -= flux * length / area;
                } else {
                    const lf::mesh::Entity *n_cell = (*neighbour_cells_p)(*cell)[i];
                    const int col = dofh.GlobalDofIndices(*n_cell)[0];
                    // TODO 4.3
                    B.coeffRef(row, col) -= flux * length / area;
                }
            } else {
                // Boundary condition for flux: no flux out!
                // TODO 4.4
                if (flux >= 0 && edge_corners.col(0)[0] > OUTLET_X) {
                    B.coeffRef(row, row) -= flux * length / area;
                }
            }
            ++i;
        }
    }

    Eigen::VectorXd mu_u(N_dofs);
    Eigen::VectorXd mu_v(N_dofs);
    for(const lf::mesh::Entity *cell: mesh_p->Entities(0)) {
        const int global_idx = dofh.GlobalDofIndices(*cell)[0];
        mu_u[global_idx] = (*uv_cell_p)(*cell)[0];
        mu_v[global_idx] = (*uv_cell_p)(*cell)[1];
    }

    // Enforce dirichlet boundary condition
    // Flag Edges on the boundary
    lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{
        lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};

    // Iterate over all cells
    for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
        for (const auto *edge : cell->SubEntities(1)) {
            if (bd_flags(*edge)) {
                Eigen::Matrix<double, 2, 2> corner_edges =
                    lf::geometry::Corners(*edge->Geometry());

                int idx = dofh.GlobalDofIndices(*cell)[0];
                if (corner_edges.col(0)[0] < INLET_X) { // Left Boundary
                    B.row(idx) *= 0;
                    if(corner_edges.col(0)[1] > 0.2 || corner_edges.col(0)[1] < -0.2) {
                        mu_u[idx] = speed;
                        mu_v[idx] = 0.0;
                    } else {
                        // TODO 4.optional play around with the value ?
                        mu_u[idx] = speed;
                        mu_v[idx] = 0.0;
                    }
                } else if(corner_edges.col(0)[0] > OUTLET_X) { // Right Boundary
//                    B.row(idx) *= 0;
//                    mu_u[idx] = 0.0;
//                    mu_v[idx] = 0.0;
                } else if (corner_edges.col(0)[1] < BOTTOM_WALL) { // Bottom Boundary
                    // TODO 4.5
                    B.row(idx) *= 0;
                    mu_u[idx] = 0.0;
                    mu_v[idx] = 0.0;
                } else if (corner_edges.col(0)[1] > TOP_WALL) { // Top Boundary
                    // TODO 4.6
                    B.row(idx) *= 0;
                    mu_u[idx] = 0.0;
                    mu_v[idx] = 0.0;
                } else { // Airfoil boundary
                    // TODO 4.7
                    B.row(idx) *= 0;
                    mu_u[idx] = 0.0;
                    mu_v[idx] = 0.0;
                }
            }
        }
    }

    // Evolution
    // TODO 4.8 write down the Butcher scheme
    Eigen::VectorXd k0_u = B * mu_u;
    Eigen::VectorXd k1_u = B * (mu_u + tau * k0_u);
    mu_u += tau * 0.5 * (k0_u + k1_u);

    Eigen::VectorXd k0_v = B * mu_v;
    Eigen::VectorXd k1_v = B * (mu_v + tau * k0_v);
    mu_v += tau * 0.5 * (k0_v + k1_v);

    for(const lf::mesh::Entity *cell: mesh_p->Entities(0)) {
        const int global_idx = dofh.GlobalDofIndices(*cell)[0];
        (*uv_cell_p)(*cell)[0] = mu_u[global_idx];
        (*uv_cell_p)(*cell)[1] = mu_v[global_idx];
    }

};

/// @brief TODO 3: Complete this function. Once completed, you can output the result
///         and output it in paraview.
/// @param mesh_p 
/// @param uv_cell_p 
/// @param uv_vertex_p 
/// @param normals_cell_p 
/// @param neighbour_cells_p 
/// @param fe_space 
/// @param tau 
/// @param vtk_writer 
/// @param step 
void projection_step(
        std::shared_ptr<lf::mesh::Mesh> mesh_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<Eigen::Vector2d>> uv_cell_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<Eigen::Vector2d>> uv_vertex_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<Eigen::Matrix<double, 2, 3>>> normals_cell_p,
        std::shared_ptr<lf::mesh::utils::CodimMeshDataSet<std::array<const lf::mesh::Entity*, 3>>> neighbour_cells_p,
        std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
        double tau,
        std::shared_ptr<lf::io::VtkWriter> vtk_writer,
        int step) {
    
    // TODO 3.1: Get the dofhandler from the the fe_space
    const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};
    // TODO 3.2: Get the number of degrees of freedom = number of vertices
    const lf::base::size_type N_dofs(dofh.NumDofs());

    const Eigen::Vector2d cell_center{1/3., 1/3.};

    // 1. Get divergence of uv on cells using gauss theorem. Move the values to vertices using
    //      volumetric averaging.
    auto div_cell_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<double>>(mesh_p, 0);
    auto div_vertex_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<double>>(mesh_p, 2);

    // The flux through every boundary is computed using the numerical central flux. 
    for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        // const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
        // TODO 3.3: Get area of the cell.
        const double area = lf::geometry::Volume(*geo_ptr);
        const Eigen::Vector2d cell_uv = (*uv_cell_p)(*cell);
        double div = 0.;
        int i = 0;
        for (auto edge : cell->SubEntities(1)) {
            const lf::geometry::Geometry *edge_geo_ptr = edge->Geometry();
            const Eigen::MatrixXd edge_corners = lf::geometry::Corners(*edge_geo_ptr);
            const double length = lf::geometry::Volume(*edge_geo_ptr);
            const Eigen::Vector2d n = (*normals_cell_p)(*cell).col(i);
            const lf::mesh::Entity *n_cell = (*neighbour_cells_p)(*cell)[i];
            if(n_cell != nullptr) {
                const Eigen::Vector2d n_cell_uv = (*uv_cell_p)(*n_cell);

                // TODO 3.4 Use the central flux to update the divergence.
                div += 0.5 * length * n.dot(n_cell_uv + cell_uv);
            } else {
                // TODO 3.5 At the boundary, we only take the flux on the cell to update the divergence.
                div += length * n.dot(cell_uv);
            }
            ++i;
        }
        (*div_cell_p)(*cell) = div / area;
    }

    cell_to_vertex(mesh_p, div_cell_p, div_vertex_p);

    // 2. Solve diffusion on the mesh: laplacian(p) / tau = div(u) 
    // 2.1 Assemble laplacian problem

    // TODO 3.6: Transfer data from the CodimMeshDataSet to an Eigenvector to later solve LSE.
    Eigen::VectorXd phi(N_dofs);
    for(const lf::mesh::Entity *vertex: mesh_p->Entities(2)) {
        const int global_idx = dofh.GlobalDofIndices(*vertex)[0];
        // 3.6 FILL ME 
        phi[global_idx] = (*div_vertex_p)(*vertex);
    }

    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // TODO 3.7 Assemble LHS of the pressure equation. LHS = -1/tau * A
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space, lf::mesh::utils::MeshFunctionConstant(-1./tau), lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

    // TODO 3.8 Impose dirichlet boundary conditions
    // Pressure is 0 at left and right boundary.
    lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{
        lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 2)};
    std::vector<std::pair<bool,double>> pressure_bdc(N_dofs, {false, 42});
    for(const lf::mesh::Entity *vertex: mesh_p->Entities(2)) {
        const lf::geometry::Geometry *geo_ptr = vertex->Geometry();
        const Eigen::Vector2d pos = lf::geometry::Corners(*geo_ptr).col(0);
        if(bd_flags(*vertex) && (pos[0] < INLET_X || pos[0] > OUTLET_X)) {
            const int global_idx = dofh.GlobalDofIndices(*vertex)[0];
            // TODO: 3.8.1 FILL ME
            pressure_bdc[global_idx] = {true, 0};
        }
    }

    // TODO: 3.8.2 Use FixFlaggedSolutionComponents to impose dirichlet boundary conditions on the
    //      system -1/tau*A = phi
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&](int gdof_idx) {
            return pressure_bdc[gdof_idx];
        },
        A, phi);

    Eigen::SparseMatrix A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    // TODO 3.9: Solve the LSE with the given solver.
    Eigen::VectorXd pressure = solver.solve(phi);
    auto pressure_vertex_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<double>>(mesh_p, 2);

    // TODO 3.10: Transfer EigenVector data to the pressure CodimMeshDataSet datastructure.
    for(const lf::mesh::Entity *vertex: mesh_p->Entities(2)) {
        const int global_idx = dofh.GlobalDofIndices(*vertex)[0];
        // FILL ME!
        (*pressure_vertex_p)(*vertex) = pressure[global_idx];
    }

    // 4. Compute gradient of pressure, evolve using explicit Euler
    // TODO 3.11: Find the gradient of the pressure. Hint: look at the documentaiton of MeshFunctionGradFE.
    lf::fe::MeshFunctionGradFE pressure_gradient(fe_space, pressure);
    for(const lf::mesh::Entity *cell: mesh_p->Entities(0)) {
        (*uv_cell_p)(*cell) -= tau * pressure_gradient(*cell, cell_center)[0];
    }

    if(step % interval == 0) {
        vtk_writer->WriteCellData("divergence_cell" + std::to_string(step), *div_cell_p);
        vtk_writer->WritePointData("divergence_point" + std::to_string(step), *div_vertex_p);
        vtk_writer->WritePointData("pressure_point" + std::to_string(step), *pressure_vertex_p);
    }

};

// TODO 1: Complete the main function.
int main() {
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    lf::io::GmshReader reader(std::move(mesh_factory),
                              CURRENT_SOURCE_DIR "/mesh_airfoil_e211.msh");
    std::shared_ptr<lf::mesh::Mesh> mesh_p = reader.mesh();


    // TODO 1.1: For each triangle, we need to know all its triangle neighbours. Understand the following datastructure.
    //      Then fill the two missing lines.

    // cell to neighbour cell
    auto neighbour_cells_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<std::array<const lf::mesh::Entity*, 3>>>(mesh_p, 0);

    // edge to neighbour cells
    lf::mesh::utils::CodimMeshDataSet<std::array<const lf::mesh::Entity *, 2>>
        aux_obj(mesh_p, 1, {nullptr, nullptr});

    for(const lf::mesh::Entity *cell: mesh_p->Entities(0)) {
        for (const lf::mesh::Entity *edge: cell->SubEntities(1)) {
            if (aux_obj(*edge)[0] == nullptr) {
                aux_obj(*edge)[0] = cell;
            } else if (aux_obj(*edge)[1] == nullptr) {
                aux_obj(*edge)[1] = cell;
            }
        }
    }

    for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
        int counter = 0;
        for (const lf::mesh::Entity *edge : cell->SubEntities(1)) {
            if (aux_obj(*edge)[0] != cell) {
                // 1.1 FILL ME!
                (*neighbour_cells_p)(*cell)[counter] = aux_obj(*edge)[0];
            } else {
                // 1.1 FILL ME!
                (*neighbour_cells_p)(*cell)[counter] = aux_obj(*edge)[1];
            }
            counter++;
        }
    }

    auto vtk_writer = std::make_shared<lf::io::VtkWriter>(mesh_p, CURRENT_SOURCE_DIR "/output.vtk");
    
    // TODO 1.2: Create a vertex centered linear FEM shared pointer. Hint: use the std::make_shared function.
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
    // TODO 1.3: Get the dofhandler of the created vertex centered space.
    const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};
    // TODO 1.4: Get the number of degrees of freedom.
    const lf::base::size_type N_dofs(dofh.NumDofs());

    // Cell centered uv values
    auto uv_cell_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<Eigen::Vector2d>>(mesh_p, 0);

    // Vertex centered uv values
    auto uv_vertex_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<Eigen::Vector2d>>(mesh_p, 2);

    // Initialize velocity field u(x,y), v(x,y).
    //      Start with arbitrary parallel lines.
    auto uv = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        return {speed, 0};
    };

    for(const lf::mesh::Entity* cell: mesh_p->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
        const Eigen::Vector2d cell_center = (corners.col(0) + corners.col(1) + corners.col(2)) / 3.;
        (*uv_cell_p)(*cell) = uv(cell_center);
    }

    // TODO 1.5: Understand the normals code. Find the corresponding formula in the script, 
    //     as computing the normals of the mesh might come to the exam.

    // Compute the triangle normals on the all the mesh
    auto normals_cell_p = std::make_shared<lf::mesh::utils::CodimMeshDataSet<Eigen::Matrix<double, 2, 3>>>(mesh_p, 0);

    for(const lf::mesh::Entity* cell: mesh_p->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
        Eigen::Matrix3d grad_helper;
        grad_helper.col(0) = Eigen::Vector3d::Ones();
        grad_helper.rightCols(2) = corners.transpose();
        const Eigen::MatrixXd grad_basis = grad_helper.inverse().bottomRows(2);
        const Eigen::MatrixXd n = -grad_basis;

        (*normals_cell_p)(*cell).col(0) = n.col(2).normalized();
        (*normals_cell_p)(*cell).col(1) = n.col(0).normalized();
        (*normals_cell_p)(*cell).col(2) = n.col(1).normalized();
    }

    vtk_writer->WriteCellData("uv_cell_0", *uv_cell_p);

    const double tau = Time / timesteps;
    for(int ts = 0; ts < timesteps; ++ts) {
        std::cout << "timestep:" << ts << "\n";

        // TODO 2: Complete cell_to_vertex function
        cell_to_vertex(mesh_p, uv_cell_p, uv_vertex_p);

        // TODO 4: Complete advection step
        advection_step(mesh_p, uv_cell_p, normals_cell_p, neighbour_cells_p, tau, vtk_writer);

        if(ts % interval  == 0) vtk_writer->WriteCellData("uv_cell_a" + std::to_string(ts), *uv_cell_p);

        cell_to_vertex(mesh_p, uv_cell_p, uv_vertex_p);
        // TODO 3: Complete projection step
        projection_step(mesh_p, uv_cell_p, uv_vertex_p, normals_cell_p, neighbour_cells_p, fe_space, tau, vtk_writer, ts);

        if(ts % interval  == 0) vtk_writer->WriteCellData("uv_cell_p" + std::to_string(ts), *uv_cell_p);

    }

    vtk_writer->WriteCellData("uv_cell_T", *uv_cell_p);

    return 0;
}
