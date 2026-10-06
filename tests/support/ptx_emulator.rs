//! A deliberately small, test-only interpreter for the embedded stage PTX.
//! Executes the real instructions (including addresses, branches, and barriers),
//! not a Rust transcription of the algorithm. This is not a CUDA/JIT emulator:
//! approximate math uses Rust f32 math, and device scheduling/copies need GPU tests.
use std::collections::HashMap;

pub enum Arg<'a> {
    Buffer(&'a str),
    U32(u32),
    F32(f32),
}

#[derive(Default)]
pub struct Machine {
    buffers: HashMap<String, (u64, usize)>,
    global: HashMap<u64, u32>,
    next: u64,
}

struct Instruction {
    guard: Option<(String, bool)>,
    op: String,
    args: Vec<String>,
}

struct Kernel {
    params: Vec<String>,
    labels: HashMap<String, usize>,
    code: Vec<Instruction>,
}

impl Kernel {
    fn parse(source: &str, name: &str) -> Self {
        let source = source
            .split(&format!(".entry {name}("))
            .nth(1)
            .expect("kernel exists");
        let (header, rest) = source.split_once('{').unwrap();
        let params = header
            .split(',')
            .map(|p| {
                p.trim()
                    .trim_end_matches(')')
                    .split_whitespace()
                    .last()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        let body = rest
            .split_once('}')
            .unwrap()
            .0
            .lines()
            .map(|line| line.split("//").next().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let mut labels = HashMap::new();
        let mut code = Vec::new();
        for statement in body.split(';') {
            let mut statement = statement.trim();
            if let Some((label, tail)) = statement.split_once(':') {
                labels.insert(label.trim().to_owned(), code.len());
                statement = tail.trim();
            }
            if statement.is_empty() || statement.starts_with('.') {
                continue;
            }
            let guard = if statement.starts_with('@') {
                let (predicate, tail) = statement.split_once(char::is_whitespace).unwrap();
                statement = tail.trim();
                Some((
                    predicate
                        .trim_start_matches('@')
                        .trim_start_matches('!')
                        .to_owned(),
                    !predicate.contains('!'),
                ))
            } else {
                None
            };
            let (op, args) = statement
                .split_once(char::is_whitespace)
                .unwrap_or((statement, ""));
            code.push(Instruction {
                guard,
                op: op.to_owned(),
                args: args
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().trim_matches(['[', ']']).to_owned())
                    .collect(),
            });
        }
        Self {
            params,
            labels,
            code,
        }
    }
}

#[derive(Default)]
struct Thread {
    registers: HashMap<String, u64>,
    pc: usize,
    exited: bool,
}

impl Thread {
    fn value(&self, name: &str) -> u64 {
        if name.starts_with('%') {
            *self
                .registers
                .get(name)
                .unwrap_or_else(|| panic!("uninitialized register {name}"))
        } else if let Some(hex) = name.strip_prefix("0f") {
            u32::from_str_radix(hex, 16).unwrap() as u64
        } else {
            name.parse()
                .unwrap_or_else(|_| panic!("unsupported operand {name}"))
        }
    }
    fn float(&self, name: &str) -> f32 {
        f32::from_bits(self.value(name) as u32)
    }

    fn until_barrier(
        &mut self,
        kernel: &Kernel,
        params: &HashMap<String, u64>,
        global: &mut HashMap<u64, u32>,
        shared: &mut HashMap<u64, u32>,
        writes: &mut HashMap<u64, usize>,
        owner: usize,
    ) {
        for _ in 0..1_000_000 {
            let ins = &kernel.code[self.pc];
            self.pc += 1;
            if let Some((predicate, expected)) = &ins.guard {
                if (self.value(predicate) != 0) != *expected {
                    continue;
                }
            }
            let a = &ins.args;
            let result = match ins.op.as_str() {
                "ret" => {
                    self.exited = true;
                    return;
                }
                "bar.sync" => return,
                "bra" => {
                    self.pc = kernel.labels[&a[0]];
                    continue;
                }
                "ld.param.u64" | "ld.param.u32" | "ld.param.f32" => params[&a[1]],
                "cvta.shared.u64" => match a[1].as_str() {
                    "sa" => 0,
                    "sb" => 4096,
                    _ => panic!("unknown shared symbol"),
                },
                "ld.global.f32" | "ld.shared.f32" => {
                    let address = self.value(&a[1]);
                    let memory = if ins.op.contains("global") {
                        &*global
                    } else {
                        &*shared
                    };
                    *memory
                        .get(&address)
                        .unwrap_or_else(|| panic!("{}: invalid address {address:#x}", ins.op))
                        as u64
                }
                "st.global.f32" | "st.shared.f32" => {
                    let address = self.value(&a[0]);
                    let value = self.value(&a[1]) as u32;
                    let memory = if ins.op.contains("global") {
                        &mut *global
                    } else {
                        &mut *shared
                    };
                    if ins.op.contains("global") {
                        assert!(
                            memory.contains_key(&address),
                            "{}: invalid address {address:#x}",
                            ins.op
                        );
                        if let Some(previous) = writes.insert(address, owner) {
                            assert_eq!(
                                previous, owner,
                                "overlapping thread writes at {address:#x}"
                            );
                        }
                    } else {
                        assert!(
                            address < 8192 && address % 4 == 0,
                            "invalid shared address {address:#x}"
                        );
                    }
                    memory.insert(address, value);
                    continue;
                }
                "mov.u32" | "mov.f32" => self.value(&a[1]),
                "add.u64" => self.value(&a[1]).wrapping_add(self.value(&a[2])),
                "add.u32" => {
                    (self.value(&a[1]) as u32).wrapping_add(self.value(&a[2]) as u32) as u64
                }
                "sub.u32" => {
                    (self.value(&a[1]) as u32).wrapping_sub(self.value(&a[2]) as u32) as u64
                }
                "mul.lo.u32" => {
                    (self.value(&a[1]) as u32).wrapping_mul(self.value(&a[2]) as u32) as u64
                }
                "mul.wide.u32" => {
                    (self.value(&a[1]) as u32 as u64) * (self.value(&a[2]) as u32 as u64)
                }
                "mad.lo.u32" => (self.value(&a[1]) as u32)
                    .wrapping_mul(self.value(&a[2]) as u32)
                    .wrapping_add(self.value(&a[3]) as u32) as u64,
                "div.u32" => self.value(&a[1]) / self.value(&a[2]),
                "rem.u32" => self.value(&a[1]) % self.value(&a[2]),
                "shl.b32" => ((self.value(&a[1]) as u32) << self.value(&a[2])) as u64,
                "shr.u32" => ((self.value(&a[1]) as u32) >> self.value(&a[2])) as u64,
                "setp.ge.u32" => (self.value(&a[1]) >= self.value(&a[2])) as u64,
                "setp.eq.u32" => (self.value(&a[1]) == self.value(&a[2])) as u64,
                "setp.eq.f32" => (self.float(&a[1]) == self.float(&a[2])) as u64,
                "setp.gt.f32" => (self.float(&a[1]) > self.float(&a[2])) as u64,
                "add.f32" => (self.float(&a[1]) + self.float(&a[2])).to_bits() as u64,
                "sub.f32" => (self.float(&a[1]) - self.float(&a[2])).to_bits() as u64,
                "mul.f32" => (self.float(&a[1]) * self.float(&a[2])).to_bits() as u64,
                "div.approx.f32" => (self.float(&a[1]) / self.float(&a[2])).to_bits() as u64,
                "neg.f32" => (-self.float(&a[1])).to_bits() as u64,
                "ex2.approx.f32" => self.float(&a[1]).exp2().to_bits() as u64,
                "sqrt.approx.f32" => self.float(&a[1]).sqrt().to_bits() as u64,
                "lg2.approx.f32" => self.float(&a[1]).log2().to_bits() as u64,
                "min.f32" => self.float(&a[1]).min(self.float(&a[2])).to_bits() as u64,
                op => panic!("unsupported instruction {op}"),
            };
            self.registers.insert(a[0].clone(), result);
        }
        panic!("PTX kernel did not reach a barrier/return");
    }
}

impl Machine {
    pub fn put(&mut self, name: &str, data: &[f32]) {
        if let Some((base, len)) = self.buffers.get(name).copied() {
            assert_eq!(data.len(), len);
            for (i, value) in data.iter().enumerate() {
                self.global.insert(base + i as u64 * 4, value.to_bits());
            }
        } else {
            // Distinct, widely separated nonzero bases catch pointer+pointer bugs.
            self.next += 0x10000000;
            let base = self.next;
            self.buffers.insert(name.to_owned(), (base, data.len()));
            for (i, value) in data.iter().enumerate() {
                self.global.insert(base + i as u64 * 4, value.to_bits());
            }
        }
    }
    pub fn zeros(&mut self, name: &str, len: usize) {
        self.put(name, &vec![0.0; len]);
    }
    pub fn get(&self, name: &str) -> Vec<f32> {
        let (base, len) = self.buffers[name];
        (0..len)
            .map(|i| f32::from_bits(self.global[&(base + i as u64 * 4)]))
            .collect()
    }
    pub fn launch(
        &mut self,
        source: &str,
        name: &str,
        args: &[Arg<'_>],
        grid: (usize, usize),
        width: usize,
    ) {
        self.launch_selected(source, name, args, grid, width, None);
    }

    /// Execute a single non-barrier thread at its real flattened launch index.
    /// Large-shape boundary probes need not execute every intervening token.
    pub fn launch_thread(
        &mut self,
        source: &str,
        name: &str,
        args: &[Arg<'_>],
        index: usize,
        width: usize,
    ) {
        self.launch_selected(source, name, args, (1, 1), width, Some(index));
    }

    fn launch_selected(
        &mut self,
        source: &str,
        name: &str,
        args: &[Arg<'_>],
        grid: (usize, usize),
        width: usize,
        selected: Option<usize>,
    ) {
        let kernel = Kernel::parse(source, name);
        assert_eq!(kernel.params.len(), args.len(), "{name} argument count");
        let params = kernel
            .params
            .iter()
            .cloned()
            .zip(args.iter().map(|arg| match arg {
                Arg::Buffer(name) => self.buffers[*name].0,
                Arg::U32(value) => *value as u64,
                Arg::F32(value) => value.to_bits() as u64,
            }))
            .collect();
        let mut writes = HashMap::new();
        for y in 0..grid.1 {
            for x in 0..grid.0 {
                let mut shared = HashMap::new();
                let mut threads: Vec<_> = (0..selected.map_or(width, |_| 1))
                    .map(|tid| Thread {
                        registers: HashMap::from([
                            ("%tid.x".into(), selected.map_or(tid, |i| i % width) as u64),
                            ("%ctaid.x".into(), selected.map_or(x, |i| i / width) as u64),
                            ("%ctaid.y".into(), y as u64),
                            ("%ntid.x".into(), width as u64),
                        ]),
                        ..Thread::default()
                    })
                    .collect();
                loop {
                    for (tid, thread) in threads
                        .iter_mut()
                        .enumerate()
                        .filter(|(_, thread)| !thread.exited)
                    {
                        let owner = (y * grid.0 + x) * width + tid;
                        thread.until_barrier(
                            &kernel,
                            &params,
                            &mut self.global,
                            &mut shared,
                            &mut writes,
                            owner,
                        );
                    }
                    if threads.iter().all(|thread| thread.exited) {
                        break;
                    }
                    assert!(
                        threads.iter().all(|thread| !thread.exited),
                        "divergent barrier in {name}"
                    );
                }
            }
        }
    }
}
