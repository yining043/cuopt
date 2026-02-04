# cuOpt `routing.Solve()` 功能分析与 pybind 接口复现能力

## `routing.Solve()` 的完整功能架构

### 1. **顶层架构**
```
routing.Solve()
  └─> solver_t::solve()
      └─> ges_solver_t::compute_ges_solution()
          └─> diverse_solver::perform_search()
```

### 2. **核心组件**

#### A. **GES Solver (Guided Ejection Search)**
- **`eject_until_feasible()`** - 强制移除节点直到解可行 ⭐ 关键功能
- **`fixed_route_loop()`** - 固定路径循环，尝试重新插入未服务节点
- **`greedy_insert()`** - 贪心插入未服务节点
- **`guided_ejection_search_loop()`** - GES 主循环，智能插入/移除节点
- **`try_squeeze_feasible()`** - 尝试挤压插入（临时违反约束后修复）
- **`squeeze_all_and_save()`** - 挤压所有节点并保存
- **`construct_feasible_solution()`** - 从零构建可行解

#### B. **Diversity Solver (种群管理)**
- **`perform_search()`** - 主搜索循环
- **`run_working_loop()`** - 工作循环
  - `populate_working_population()` - 填充工作种群
  - `improve_population()` - 改进种群（局部搜索）
  - `adjust_weights()` - 自适应权重调整 ⭐
  - `add_working_to_reserve()` - 添加到保留种群
  - `run_make_feasible()` - 自动修复可行性 ⭐
- **`generate_solution()`** - 生成初始解
- **`refill_reserve()`** - 重新填充保留种群

#### C. **Local Search (局部搜索)**
- **`perform_vrp_search()`** - VRP 移动搜索
- **`run_two_opt_search()`** - 2-opt 搜索
- **`run_sliding_search()`** - 滑动窗口搜索
- **`run_cycle_finder()`** - 负循环检测
- **`run_random_local_search()`** - 随机局部搜索
- **`improve()`** - 改进解
- **`perturbate()`** - 扰动解

#### D. **可行性保证机制** ⭐⭐⭐
1. **自动修复**：`if (!reserve_population.is_feasible()) { run_make_feasible(); }`
2. **`make_feasible()` 流程**：
   ```
   eject_until_feasible()  // 1. 移除不可行节点
   populate_ep_with_unserved()  // 2. 收集未服务节点
   run_random_local_search()  // 3. 随机局部搜索
   fixed_route_loop()  // 4. GES 重新插入
   try_squeeze_breaks_feasible()  // 5. 尝试重新插入 breaks
   ```
3. **优先返回可行解**：`best_feasible()` 优先于 `best()`

#### E. **自适应权重调整** ⭐
- **`adjust_weights()`** - 根据可行性自动调整权重
  - 如果解不可行：增加权重
  - 如果解可行：减少权重
- **初始权重**：`[10000, 10000, 100, 1000, 1000, 1000, ...]` (capacity=100)

## pybind 接口 (`VrpLS`) 提供的功能

### ✅ **已支持的功能**
1. **Local Search 操作**：
   - `perform_vrp_search()` ✅
   - `run_two_opt_search()` ✅
   - `run_sliding_search()` ✅
   - `run_cycle_finder()` ✅

2. **资源管理**：
   - `acquire_resource()` ✅
   - `release_resource()` ✅
   - `sync_streams()` ✅

3. **节点管理**：
   - `extract_nodes_to_search()` ✅
   - `restore_found_nodes()` ✅
   - `sample_nodes_to_search()` ✅

4. **权重管理**：
   - `get_weights()` ✅
   - `set_weights()` ✅
   - `get_selection_weights()` ✅
   - `set_selection_weights()` ✅

5. **解管理**：
   - `get_solution_routes()` ✅
   - `get_cost()` ✅
   - `initialize_search()` ✅

### ❌ **缺失的关键功能**

1. **GES 核心功能**：
   - ❌ `eject_until_feasible()` - **无法强制移除节点**
   - ❌ `fixed_route_loop()` - **无法重新插入未服务节点**
   - ❌ `greedy_insert()` - **无法贪心插入**
   - ❌ `guided_ejection_search_loop()` - **无法执行 GES 循环**
   - ❌ `try_squeeze_feasible()` - **无法挤压插入**

2. **种群管理**：
   - ❌ 多解管理（reserve_population, working_population）
   - ❌ 多样性管理
   - ❌ 重组操作

3. **自动修复机制**：
   - ❌ `make_feasible()` - **无法自动修复可行性**
   - ❌ `run_make_feasible()` - **无法自动调用修复**

4. **解生成**：
   - ❌ `generate_solution()` - **无法从零生成解**
   - ❌ `construct_feasible_solution()` - **无法构建可行解**

5. **高级搜索**：
   - ❌ `run_random_local_search()` - **无法随机局部搜索**
   - ❌ `improve()` - **无法改进解**
   - ❌ `perturbate()` - **无法扰动解**

## 能否用 pybind 接口复现？

### ✅ **可以部分复现的功能**

1. **局部搜索流程**：
   ```python
   # 可以复现 run_best_local_search 的逻辑
   cuopt_env.perform_vrp_search()
   cuopt_env.run_sliding_search()
   cuopt_env.run_two_opt_search()
   cuopt_env.run_cycle_finder()
   ```

2. **手动权重调整**：
   ```python
   # 可以手动实现 adjust_weights 的逻辑
   weights = cuopt_env.get_weights()
   if not is_feasible:
       weights[2] *= 1.5  # 增加 capacity 权重
       cuopt_env.set_weights(weights)
   ```

3. **手动可行性检查**：
   ```python
   # 可以手动检查可行性
   routes = cuopt_env.get_solution_routes()
   is_feasible = validate_solution_feasibility(routes, vrp_instance)
   ```

### ❌ **无法复现的关键功能**

1. **`eject_until_feasible()`**：
   - **问题**：pybind 接口没有提供移除节点的功能
   - **影响**：无法强制修复不可行解
   - **解决方案**：需要在 C++ 层面添加此功能

2. **`fixed_route_loop()` / GES**：
   - **问题**：pybind 接口没有 GES 相关功能
   - **影响**：无法重新插入未服务节点
   - **解决方案**：需要暴露 GES 接口

3. **自动修复机制**：
   - **问题**：没有 `make_feasible()` 接口
   - **影响**：无法自动保证可行性
   - **解决方案**：需要暴露 `make_feasible()` 接口

## 建议的实现方案

### 方案 1：扩展 pybind 接口（推荐）

在 `VrpLS` 类中添加以下方法：

```cpp
// 可行性修复
bool eject_until_feasible(bool add_slack_to_sol = true);
bool make_feasible(double time_limit, const std::vector<double>& weights);

// GES 功能
bool fixed_route_loop();
bool greedy_insert(bool insert_all = false);
bool guided_ejection_search_loop(int counter, bool minimize_routes);

// 解生成
bool generate_solution(const std::vector<int>& target_vehicle_ids, 
                       double time_limit, 
                       const std::vector<double>& weights);
bool construct_feasible_solution();

// 高级搜索
void run_random_local_search(bool include_objective = false);
void improve(const std::vector<double>& weights, double time_limit);
void perturbate(const std::vector<double>& weights, int perturbation_count);

// 可行性检查
bool is_feasible() const;
void populate_ep_with_unserved();  // 填充 ejection pool
```

### 方案 2：在 Python 层面实现部分功能

对于无法直接访问的功能，可以在 Python 层面实现简化版本：

```python
def manual_eject_until_feasible(cuopt_env, vrp_instance):
    """手动实现 eject_until_feasible 的简化版本"""
    routes = cuopt_env.get_solution_routes()
    demands = vrp_instance['demands']
    capacity = vrp_instance['vehicle_capacity']
    
    # 找出不可行的路径
    infeasible_routes = []
    for i, route in enumerate(routes):
        route_load = sum(demands[node] for node in route if node != 0)
        if route_load > capacity:
            infeasible_routes.append(i)
    
    # 移除节点直到可行（简化版本）
    # 注意：这需要能够修改解，但 pybind 接口可能不支持
    # 需要重新初始化搜索
    ...
```

### 方案 3：直接使用 `routing.Solve()`

如果只需要可行解，直接使用 cuOpt 的完整求解器：

```python
import cuopt.routing as routing

data_model = routing.DataModel(...)
solver_settings = routing.SolverSettings()
solution = routing.Solve(data_model, solver_settings)  # 自动保证可行性
```

## 总结

| 功能类别 | 完整求解器 | pybind 接口 | 复现难度 |
|---------|-----------|------------|---------|
| 局部搜索 | ✅ | ✅ | ✅ 容易 |
| 权重调整 | ✅ 自动 | ✅ 手动 | ✅ 容易 |
| 可行性修复 | ✅ 自动 | ❌ | ❌ 困难 |
| GES 功能 | ✅ | ❌ | ❌ 困难 |
| 种群管理 | ✅ | ❌ | ❌ 困难 |
| 解生成 | ✅ | ❌ | ❌ 困难 |

**结论**：
- ✅ **可以复现**：局部搜索流程、手动权重调整、可行性检查
- ❌ **无法复现**：自动可行性修复、GES 功能、种群管理
- 💡 **建议**：如果需要可行性保证，优先考虑扩展 pybind 接口或直接使用 `routing.Solve()`

