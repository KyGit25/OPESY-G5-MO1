/*
G5 - PROCESS SCHEDULER AND CLI (DESAMITO, MARISTELA, SANLUIS)
*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <deque>
#include <stack>
#include <memory>    
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>
#include <chrono>
#include <iomanip>      
#include <algorithm>
#include <cmath>

/**
 * @brief Defines the config params read from config.txt
 */
struct Config {
    int num_cpu = 0;
    std::string scheduler = "rr";
    int quantum_cycles = 1;
    int batch_process_freq = 0;
    int min_ins = 0;
    int max_ins = 0;
    int delays_per_exec = 0;
};

/**
 * @brief Types of instructions for the barebones interpreter.
 * FOR_START and FOR_END are used to implement the FOR loop logic.
 */
enum class InstructionType {
    PRINT,
    DECLARE,
    ADD,
    SUBTRACT,
    SLEEP,
    FOR_START,
    FOR_END
};

/**
 * @brief Represents a single barebones instruction.
 */
struct Instruction {
    InstructionType type;
    std::vector<std::string> args;
};

/**
 * @brief Represents the state of a process.
 */
enum class ProcessState {
    READY,
    RUNNING,
    SLEEPING,
    TERMINATED
};

/**
 * @brief Information needed to manage a FOR loop's execution.
 */
struct ForLoopInfo {
    int start_pc;             // The PC of the FOR_START instruction + 1
    int remaining_iterations;
};

/**
 * @brief Represents a single process in the emulator.
 */
struct Process {
    int id;
    std::string name;
    ProcessState state;
    std::vector<Instruction> instructions;
    int pc = 0; // Program Counter
    std::map<std::string, uint16_t> variables;
    std::stack<ForLoopInfo> for_loop_stack;

    // Execution & Scheduling
    uint64_t wake_up_tick = 0; // Tick at which this process wakes from SLEEP
    int core_id = -1;            // CPU core it's running on (-1 if not running)
    int total_instructions = 0;
    int executed_instructions = 0;

    // Logging & Reporting
    std::string start_time_str;
    std::vector<std::string> logs;
    std::mutex log_mutex; // Protects the 'logs' vector

    Process(int id, std::string name, int total_ins)
        : id(id), name(std::move(name)), state(ProcessState::READY), 
          total_instructions(total_ins) {
        
        // Get and format the start time
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%m/%d/%Y %H:%M:%S");
        start_time_str = ss.str();
    }
};

// Configuration
Config global_config;

// Emulator State
std::atomic<bool> os_running(true);
std::atomic<bool> initialized(false);
std::atomic<bool> scheduler_running(false); // For scheduler-start/stop
std::atomic<uint64_t> cpu_ticks(0);
std::atomic<int> process_id_counter(0);

// Process Lists 
std::vector<std::shared_ptr<Process>> all_processes;
std::mutex all_processes_mutex;

std::deque<std::shared_ptr<Process>> ready_queue;
std::mutex ready_queue_mutex;
std::condition_variable ready_queue_cv;

std::vector<std::shared_ptr<Process>> sleeping_processes;
std::mutex sleeping_processes_mutex;

std::vector<std::shared_ptr<Process>> finished_processes;
std::mutex finished_processes_mutex;

// CPU Core State
std::vector<std::shared_ptr<Process>> current_core_processes; // [core_id] -> Process
std::mutex current_core_processes_mutex;
// Using unique_ptr to manage atomic variables since they're not copyable/movable
std::vector<std::unique_ptr<std::atomic<bool>>> core_busy_status; // For utilization report

// Thread Management
std::vector<std::thread> cpu_core_threads;
std::thread manager_thread;

// Random Number Generator (for process generation)
std::mt19937 rng;

void main_cli_loop();
bool load_config(const std::string& filename);
std::shared_ptr<Process> generate_dummy_process();
void cpu_worker_function(int core_id);
void manager_thread_function();
void execute_instruction(const std::shared_ptr<Process>& process, const Instruction& instr, bool& jumped, bool& process_slept);
void display_report(bool write_to_file);
void enter_screen_mode(const std::shared_ptr<Process>& process);
std::vector<std::string> split_string(const std::string& s, char delimiter);
int64_t parse_value(const std::shared_ptr<Process>& process, const std::string& arg);
std::string get_current_timestamp_log();

/**
 * @brief Splits a string by a delimiter.
 */
std::vector<std::string> split_string(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief Gets a timestamp string for process logs.
 */
std::string get_current_timestamp_log() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    // Format: (MM/DD/YYYY HH:MM:SS)
    ss << "(" << std::put_time(std::localtime(&in_time_t), "%m/%d/%Y %H:%M:%S") << ")";
    return ss.str();
}

/**
 * @brief Parses an argument, which could be a literal number or a variable name.
 * @return The value of the argument. Returns 0 if variable not found.
 */
int64_t parse_value(const std::shared_ptr<Process>& process, const std::string& arg) {
    if (arg.empty()) return 0;

    // Check if it's a variable
    if (arg[0] == '$') {
        std::string var_name = arg.substr(1);
        if (process->variables.count(var_name)) {
            return process->variables[var_name];
        } else {
            return 0;
        }
    }

    // Otherwise, it's a literal
    try {
        return std::stoll(arg);
    } catch (...) {
        return 0; // Failed to parse
    }
}

/**
 * @brief Clamps a value to the uint16_t range (0 to 65535).
 */
uint16_t clamp_uint16(int64_t value) {
    if (value < 0) return 0;
    if (value > 65535) return 65535; // UINT16_MAX
    return static_cast<uint16_t>(value);
}

/**
 * @brief Generates a new dummy process with a random set of instructions.
 */
std::shared_ptr<Process> generate_dummy_process() {
    // Initialize random distributions
    std::uniform_int_distribution<int> dist_ins(global_config.min_ins, global_config.max_ins);
    std::uniform_int_distribution<int> dist_op(0, 5); // Opcode type
    std::uniform_int_distribution<int> dist_val(1, 100); // For repeats, sleep
    std::uniform_int_distribution<int> dist_var(0, 2); // For var names x, y, z

    int num_instructions = dist_ins(rng);
    int current_id = process_id_counter.fetch_add(1);
    std::string process_name = "p" + std::to_string(current_id);
    
    // Pad name to p001, p002, etc. 
    std::stringstream ss_name;
    ss_name << "p" << std::setfill('0') << std::setw(3) << current_id;
    process_name = ss_name.str();


    auto process = std::make_shared<Process>(current_id, process_name, num_instructions);

    int nest_level = 0;
    const int max_nest_level = 3;
    std::vector<std::string> var_names = {"x", "y", "z"};

    for (int i = 0; i < num_instructions; ++i) {
        Instruction instr;
        int op = dist_op(rng);

        // Try to close loops if we are near the end
        if (nest_level > 0 && i >= num_instructions - nest_level) {
            op = 6; // Force FOR_END
        }

        switch (op) {
            case 0: // PRINT
                instr.type = InstructionType::PRINT;
                instr.args.push_back("\"Hello world from *$process_name*!\"");
                break;
            
            case 1: // DECLARE
                instr.type = InstructionType::DECLARE;
                instr.args.push_back("$" + var_names[dist_var(rng)]);
                instr.args.push_back(std::to_string(dist_val(rng)));
                break;

            case 2: // ADD
                instr.type = InstructionType::ADD;
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var1
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var2
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var3 (result)
                break;

            case 3: // SUBTRACT
                instr.type = InstructionType::SUBTRACT;
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var1
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var2
                instr.args.push_back("$" + var_names[dist_var(rng)]); // var3 (result)
                break;

            case 4: // SLEEP
                instr.type = InstructionType::SLEEP;
                instr.args.push_back(std::to_string(dist_val(rng) % 10 + 1)); // Sleep 1-10 ticks
                break;

            case 5: // FOR_START
                if (nest_level < max_nest_level) {
                    instr.type = InstructionType::FOR_START;
                    instr.args.push_back(std::to_string(dist_val(rng) % 5 + 1)); // Repeats 1-5 times
                    process->instructions.push_back(instr);
                    nest_level++;
                } else {
                    // Can't nest, just do a print instead
                    instr.type = InstructionType::PRINT;
                    instr.args.push_back("\"Max nest reached!\"");
                }
                break;

            case 6: // FOR_END (Forced)
            default: // FOR_END (Random)
                if (nest_level > 0) {
                    instr.type = InstructionType::FOR_END;
                    process->instructions.push_back(instr);
                    nest_level--;
                } else {
                    // No loop to end, just do a print
                    instr.type = InstructionType::PRINT;
                    instr.args.push_back("\"No loop to end!\"");
                }
                break;
        }

        // Add the instruction (unless it was a FOR loop)
        if (op < 5 || (op == 5 && nest_level <= max_nest_level) || (op >= 6 && nest_level >= 0)) {
            if(instr.type != InstructionType::FOR_START && instr.type != InstructionType::FOR_END) {
                 process->instructions.push_back(instr);
            }
        }
    }

    // Ensure all loops are closed
    while (nest_level > 0) {
        Instruction instr;
        instr.type = InstructionType::FOR_END;
        process->instructions.push_back(instr);
        nest_level--;
    }
    
    // Update total instructions to the actual number generated
    process->total_instructions = process->instructions.size();

    return process;
}

/**
 * @brief Executes a single barebones instruction for a given process.
 *
 * @param process The process executing the instruction.
 * @param instr The instruction to execute.
 * @param jumped Output param: set to true if the PC was modified (i.e., a FOR loop)
 * @param process_slept Output param: set to true if the instruction was SLEEP
 */
void execute_instruction(
    const std::shared_ptr<Process>& process,
    const Instruction& instr,
    bool& jumped,
    bool& process_slept) 
{
    jumped = false;
    process_slept = false;

    try {
        switch (instr.type) {
            case InstructionType::PRINT: {
                // Arg0: The message string, e.g., "$var" or "\"literal msg\""
                std::string msg = instr.args[0];
                std::string output;
                
                if (msg.front() == '$') {
                    // Print variable
                    output = std::to_string(parse_value(process, msg));
                } else {
                    // Print literal string
                    output = msg;
                    if (output.front() == '"' && output.back() == '"') {
                        output = output.substr(1, output.length() - 2); // Remove quotes
                    }
                    
                    // Replace *$process_name* with the actual name
                    std::string placeholder = "*$process_name*";
                    size_t pos = output.find(placeholder);
                    if (pos != std::string::npos) {
                        output.replace(pos, placeholder.length(), process->name);
                    }
                }

                std::string log_entry = get_current_timestamp_log() + " Core:" + 
                                        std::to_string(process->core_id) + " " + output;
                
                std::lock_guard<std::mutex> lock(process->log_mutex);
                process->logs.push_back(log_entry);
                break;
            }

            case InstructionType::DECLARE: {
                // Arg0: var_name (e.g., "$x"), Arg1: value (e.g., "100" or "$y")
                std::string var_name = instr.args[0].substr(1); // Remove '$'
                uint16_t value = clamp_uint16(parse_value(process, instr.args[1]));
                process->variables[var_name] = value;
                break;
            }

            case InstructionType::ADD: {
                // Arg0: var1, Arg1: var2, Arg2: result_var
                int64_t val1 = parse_value(process, instr.args[0]);
                int64_t val2 = parse_value(process, instr.args[1]);
                std::string result_var = instr.args[2].substr(1); // Remove '$'
                process->variables[result_var] = clamp_uint16(val1 + val2);
                break;
            }

            case InstructionType::SUBTRACT: {
                // Arg0: var1, Arg1: var2, Arg2: result_var
                int64_t val1 = parse_value(process, instr.args[0]);
                int64_t val2 = parse_value(process, instr.args[1]);
                std::string result_var = instr.args[2].substr(1); // Remove '$'
                process->variables[result_var] = clamp_uint16(val1 - val2);
                break;
            }

            case InstructionType::SLEEP: {
                // Arg0: ticks
                int64_t sleep_ticks = parse_value(process, instr.args[0]);
                process->state = ProcessState::SLEEPING;
                // Add current_tick + sleep_ticks.
                // We use +1 because the current tick is already in progress.
                process->wake_up_tick = cpu_ticks.load() + sleep_ticks + 1;
                process_slept = true;
                break;
            }

            case InstructionType::FOR_START: {
                // Arg0: repeats
                int iterations = static_cast<int>(parse_value(process, instr.args[0]));
                // Push the *next* PC, which is the start of the loop body
                process->for_loop_stack.push({process->pc + 1, iterations});
                break;
            }

            case InstructionType::FOR_END: {
                if (process->for_loop_stack.empty()) {
                    // Malformed code, just continue
                    break;
                }
                
                ForLoopInfo& loop = process->for_loop_stack.top();
                loop.remaining_iterations--;

                if (loop.remaining_iterations > 0) {
                    // Jump back to the start of the loop body
                    process->pc = loop.start_pc;
                    jumped = true;
                } else {
                    // Loop finished, pop it
                    process->for_loop_stack.pop();
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        // Handle potential errors, e.g., out-of-bounds args
        std::string log_entry = get_current_timestamp_log() + " Core:" + 
                                std::to_string(process->core_id) + " ERROR executing instruction: " + e.what();
        std::lock_guard<std::mutex> lock(process->log_mutex);
        process->logs.push_back(log_entry);
    }
}

/**
 * @brief The function executed by each CPU Core thread.
 *
 * This function loops, waiting for processes on the ready_queue.
 * When a process is available, it "executes" it for a time slice
 * (if RR) or until completion/sleep (if FCFS).
 */
void cpu_worker_function(int core_id) {
    while (os_running) {
        std::shared_ptr<Process> process;

        // 1. Get a process from the ready queue
        {
            std::unique_lock<std::mutex> lock(ready_queue_mutex);
            // Wait until the queue is not empty OR the OS is shutting down
            ready_queue_cv.wait(lock, [&] {
                return !ready_queue.empty() || !os_running;
            });

            if (!os_running) {
                return; // OS is shutting down
            }

            // Get the process
            if (global_config.scheduler == "fcfs") {
                process = ready_queue.front();
                ready_queue.pop_front();
            } else { // "rr"
                process = ready_queue.front();
                ready_queue.pop_front();
            }
            
            // Mark this core as busy and running the process
            core_busy_status[core_id]->store(true);
            process->state = ProcessState::RUNNING;
            process->core_id = core_id;

            {
                std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);
                current_core_processes[core_id] = process;
            }
        } // Release ready_queue_mutex

        // 2. Execute the process
        bool process_finished = false;
        bool process_slept = false;
        
        // FCFS runs until block/terminate, RR runs for a quantum
        int cycles_to_run = (global_config.scheduler == "rr")
                                ? global_config.quantum_cycles
                                : process->total_instructions; // Effectively "run to completion"

        for (int i = 0; i < cycles_to_run; ++i) {
            if (process->pc >= process->instructions.size()) {
                process_finished = true;
                break;
            }

            const Instruction& instr = process->instructions[process->pc];
            bool jumped = false;
            execute_instruction(process, instr, jumped, process_slept);
            
            process->executed_instructions++;

            if (!jumped) {
                process->pc++;
            }

            // Simulate the instruction delay as a busy-wait
            if (global_config.delays_per_exec > 0) {
                uint64_t start_tick = cpu_ticks.load();
                uint64_t end_tick = start_tick + global_config.delays_per_exec;
                while (cpu_ticks.load() < end_tick && os_running) {
                    std::this_thread::yield(); 
                }
            } else {
                // "If zero, each instruction is executed per CPU cycle."
                uint64_t current_tick = cpu_ticks.load();
                while (cpu_ticks.load() == current_tick && os_running) {
                     std::this_thread::yield();
                }
            }

            if (process_slept || !os_running) {
                break; // Instruction was SLEEP or OS is stopping
            }
        }

        // 3. Post-execution: Update process state and lists
        {
            // Mark core as idle
            std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);
            current_core_processes[core_id] = nullptr;
        }
        process->core_id = -1;
        core_busy_status[core_id]->store(false);


        if (!os_running) {
            // If OS is stopping, put process back in ready queue
            process->state = ProcessState::READY;
            std::lock_guard<std::mutex> lock(ready_queue_mutex);
            ready_queue.push_front(process); // Put back for clean shutdown
            return;
        }

        if (process_finished || process->pc >= process->instructions.size()) {
            process->state = ProcessState::TERMINATED;
            std::lock_guard<std::mutex> lock(finished_processes_mutex);
            finished_processes.push_back(process);
        } else if (process_slept) {
            // State was already set to SLEEPING
            std::lock_guard<std::mutex> lock(sleeping_processes_mutex);
            sleeping_processes.push_back(process);
        } else {
            // Quantum expired (for RR)
            process->state = ProcessState::READY;
            std::lock_guard<std::mutex> lock(ready_queue_mutex);
            ready_queue.push_back(process); // Put at the back
        }
    }
}

/**
 * @brief The function for the single manager thread.
 *
 * This thread is the heartbeat of the OS. It:
 * 1. Increments cpu_ticks.
 * 2. Wakes up processes from the sleeping_processes list.
 * 3. Generates new batch processes if scheduler_running is true.
 */
void manager_thread_function() {
    uint64_t next_generation_tick = 0;
    
    if (global_config.batch_process_freq > 0) {
        next_generation_tick = global_config.batch_process_freq;
    } else {
        // Avoid divide-by-zero, just set it far in the future
        next_generation_tick = (uint64_t)-1;
    }

    while (os_running) {
        // 1. Sleep for 1ms to simulate one tick passing
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // 2. Increment the global CPU tick
        uint64_t current_tick = cpu_ticks.fetch_add(1) + 1;

        // 3. Check sleeping processes
        {
            std::lock_guard<std::mutex> lock(sleeping_processes_mutex);
            // Iterate and erase idiom
            sleeping_processes.erase(
                std::remove_if(sleeping_processes.begin(), sleeping_processes.end(),
                    [&](const std::shared_ptr<Process>& p) {
                        if (p->wake_up_tick <= current_tick) {
                            p->state = ProcessState::READY;
                            std::lock_guard<std::mutex> ready_lock(ready_queue_mutex);
                            ready_queue.push_back(p);
                            return true; // Remove from sleeping list
                        }
                        return false; // Keep sleeping
                    }),
                sleeping_processes.end());
        }

        // 4. Check if we need to generate new processes
        if (scheduler_running && current_tick >= next_generation_tick) {
            if(global_config.batch_process_freq > 0) {
                 // Generate a new process
                std::shared_ptr<Process> new_process = generate_dummy_process();

                // Add to all_processes list
                {
                    std::lock_guard<std::mutex> lock(all_processes_mutex);
                    all_processes.push_back(new_process);
                }
                // Add to ready queue
                {
                    std::lock_guard<std::mutex> lock(ready_queue_mutex);
                    ready_queue.push_back(new_process);
                }
                
                // Set the next generation time
                next_generation_tick = current_tick + global_config.batch_process_freq;
            }
        }
        
        // 5. Notify any waiting CPU cores
        ready_queue_cv.notify_all();
    }
}

/**
 * @brief Prints the report for 'screen -ls' or 'report-util'.
 */
void display_report(bool write_to_file) {
    std::stringstream ss;

    // 1. Calculate CPU Utilization
    int busy_cores = 0;
    for (size_t i = 0; i < global_config.num_cpu; ++i) {
        if (core_busy_status[i]->load()) {
            busy_cores++;
        }
    }
    double utilization = (global_config.num_cpu > 0)
                           ? (static_cast<double>(busy_cores) / global_config.num_cpu) * 100.0
                           : 0.0;

    ss << "CPU utilization: " << std::fixed << std::setprecision(0) << utilization << "%\n";
    ss << "Cores used: " << busy_cores << "\n";
    ss << "Cores available: " << global_config.num_cpu - busy_cores << "\n\n";

    // Lock all process lists needed for the report
    std::lock_guard<std::mutex> all_lock(all_processes_mutex);
    std::lock_guard<std::mutex> finished_lock(finished_processes_mutex);
    std::lock_guard<std::mutex> ccp_lock(current_core_processes_mutex);

    // 2. Running Processes
    ss << "Running processes:\n";
    for (int i = 0; i < global_config.num_cpu; ++i) {
        auto process = current_core_processes[i];
        if (process) {
            ss << "  " << process->name << "  "
               << "(" << process->start_time_str << ")  Core: " << i << "   "
               << process->executed_instructions << " / " << process->total_instructions << "\n";
        }
    }

    // 3. Finished Processes
    ss << "\nFinished processes:\n";
    for (const auto& process : finished_processes) {
        ss << "  " << process->name << "  "
           << "(" << process->start_time_str << ")  Finished  "
           << process->executed_instructions << " / " << process->total_instructions << "\n";
    }

    // 4. Output
    std::cout << ss.str();
    if (write_to_file) {
        std::ofstream out_file("csopesy-log.txt");
        out_file << ss.str();
        out_file.close();
        std::cout << "\nReport generated at ./csopesy-log.txt\n";
    }
}

/**
 * @brief Enters the interactive "screen" for a specific process.
 */
void enter_screen_mode(const std::shared_ptr<Process>& process) {
    if (!process) {
        std::cout << "Error: Process not found.\n";
        return;
    }

    std::string line;
    while (os_running) {
        // Clear screen (crude, but functional)
        std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
        
        std::cout << "--- Process Screen: " << process->name << " ---\n";
        std::cout << "ID:           " << process->id << "\n";
        std::cout << "Status:       ";
        switch (process->state) {
            case ProcessState::READY: std::cout << "Ready\n"; break;
            case ProcessState::RUNNING: std::cout << "Running (Core " << process->core_id << ")\n"; break;
            case ProcessState::SLEEPING: std::cout << "Sleeping (wake at " << process->wake_up_tick << ")\n"; break;
            case ProcessState::TERMINATED: std::cout << "Terminated\n"; break;
        }
        std::cout << "Instructions: " << process->executed_instructions << " / " << process->total_instructions << "\n";
        std::cout << "PC:           " << process->pc << "\n";
        
        std::cout << "\n--- Variables ---\n";
        if (process->variables.empty()) {
            std::cout << "(No variables declared)\n";
        }
        for (const auto& pair : process->variables) {
            std::cout << "$" << pair.first << " = " << pair.second << "\n";
        }

        std::cout << "\n--- Logs ---\n";
        {
            std::lock_guard<std::mutex> lock(process->log_mutex);
            if (process->logs.empty()) {
                std::cout << "(No log output)\n";
            }
            // Print only the last 15 logs for readability
            int start = std::max(0, static_cast<int>(process->logs.size()) - 15);
            for (size_t i = start; i < process->logs.size(); ++i) {
                std::cout << process->logs[i] << "\n";
            }
        }
        
        if (process->state == ProcessState::TERMINATED) {
            std::cout << "\n--- FINISHED ---";
        }

        std::cout << "\n\nType 'exit' to return to main menu: ";
        
        // This is a simple blocking poll.
        if (!std::getline(std::cin, line)) {
            break; // EOF
        }
        if (line == "exit") {
            break;
        }
    }
}

/**
 * @brief Finds a process by name (e.g., "p001")
 */
std::shared_ptr<Process> find_process_by_name(const std::string& name) {
    std::lock_guard<std::mutex> lock(all_processes_mutex);
    auto it = std::find_if(all_processes.begin(), all_processes.end(), 
        [&name](const std::shared_ptr<Process>& p) {
        return p->name == name;
    });

    if (it != all_processes.end()) {
        return *it;
    }
    
    // Check finished processes too
    std::lock_guard<std::mutex> finished_lock(finished_processes_mutex);
    it = std::find_if(finished_processes.begin(), finished_processes.end(), 
        [&name](const std::shared_ptr<Process>& p) {
        return p->name == name;
    });
    
    if (it != finished_processes.end()) {
        return *it;
    }

    return nullptr;
}


/**
 * @brief Loads configuration from "config.txt".
 */
bool load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return false;
    }

    std::string line, key, value;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> key >> value)) { continue; }

        try {
            if (key == "num-cpu") global_config.num_cpu = std::stoi(value);
            else if (key == "scheduler") global_config.scheduler = value;
            else if (key == "quantum-cycles") global_config.quantum_cycles = std::stoi(value);
            else if (key == "batch-process-freq") global_config.batch_process_freq = std::stoi(value);
            else if (key == "min-ins") global_config.min_ins = std::stoi(value);
            else if (key == "max-ins") global_config.max_ins = std::stoi(value);
            else if (key == "delays-per-exec") global_config.delays_per_exec = std::stoi(value);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing config line: " << line << "\n";
            return false;
        }
    }

    // Validate config
    if (global_config.num_cpu <= 0) {
        std::cerr << "Error: num-cpu must be > 0.\n"; return false;
    }
    
    // Initialize RNG
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    
    return true;
}

int main() {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Welcome to CSOPESY Emulator!\n";
    std::cout << "\nDevelopers:\n";
    std::cout << " - Desamito, Hector Francis Seigmund\n";
    std::cout << " - Maristela, Kyle Gabriel\n";
    std::cout << " - San Luis, Owen Philip\n";
    std::cout << "\nLast Updated: " << __DATE__ << " " << __TIME__ << "\n";
    std::cout << "----------------------------------------\n";

    std::string line;
    while (os_running) {
        std::cout << "\nroot:\\> ";
        if (!std::getline(std::cin, line)) {
            if (os_running) {
                line = "exit"; // Handle Ctrl+D or EOF
            } else {
                break;
            }
        }

        std::vector<std::string> tokens = split_string(line, ' ');
        if (tokens.empty()) continue;

        std::string command = tokens[0];

        if (command == "initialize") {
            if (initialized) {
                std::cout << "Emulator already initialized.\n";
                continue;
            }

            if (load_config("config.txt")) {
                std::cout << "Configuration loaded. Starting " << global_config.num_cpu << " cores.\n";
                
                // Resize core state vectors
                current_core_processes.resize(global_config.num_cpu);
                core_busy_status.resize(global_config.num_cpu);
                for (int i = 0; i < global_config.num_cpu; ++i) {
                    core_busy_status[i] = std::make_unique<std::atomic<bool>>(false);
                }

                // Start CPU core threads
                for (int i = 0; i < global_config.num_cpu; ++i) {
                    cpu_core_threads.emplace_back(cpu_worker_function, i);
                }
                
                // Start the manager thread
                manager_thread = std::thread(manager_thread_function);

                initialized = true;
                std::cout << "Emulator initialized.\n";
            } else {
                std::cout << "Failed to initialize. Check config.txt.\n";
            }
            continue;
        }

        // All other commands require initialization
        if (!initialized) {
            std::cout << "Error: You must run 'initialize' first.\n";
            continue;
        }

        if (command == "exit") {
            os_running = false;
            ready_queue_cv.notify_all(); // Wake all threads
            
            // Join all threads
            manager_thread.join();
            for (auto& t : cpu_core_threads) {
                t.join();
            }
            
            std::cout << "Emulator terminating.\n";
            break;

        } else if (command == "scheduler-start") {
            scheduler_running = true;
            std::cout << "Batch process generator started.\n";

        } else if (command == "scheduler-stop") {
            scheduler_running = false;
            std::cout << "Batch process generator stopped.\n";

        } else if (command == "report-util") {
            display_report(true);

        } else if (command == "screen") {
            if (tokens.size() < 2) {
                std::cout << "Usage: screen [-ls | -s <process_name> | -r <process_name>]\n";
            } else if (tokens[1] == "-ls") {
                display_report(false);
            } else if (tokens[1] == "-s" || tokens[1] == "-r") {
                if (tokens.size() < 3) {
                    std::cout << "Usage: " << tokens[1] << " <process_name>\n";
                } else {
                    auto p = find_process_by_name(tokens[2]);
                    if (p) {
                        enter_screen_mode(p);
                    } else {
                        std::cout << "Process <" << tokens[2] << "> not found.\n";
                    }
                }
            } else {
                std::cout << "Unknown 'screen' command.\n";
            }
        } else {
            std::cout << "Unknown command: " << command << "\n";
        }
    }

    return 0;
}
