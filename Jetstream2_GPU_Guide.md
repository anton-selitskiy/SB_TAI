# ESE 670 — Using Jetstream2 GPUs

Our course has free GPU time on **Jetstream2** through NSF ACCESS (allocation **CIS262025**).
Each student runs their own GPU virtual machine (VM) and works in JupyterLab from their laptop.

> **The one rule:** when you finish working, **shelve** your VM (step 7).
> A running VM uses our shared allocation even when nobody is using it.

---

## 0. Log in to Exosphere

1. Go to **<https://jetstream2.exosphere.app>** and sign in with your **ACCESS** account.
   Official guide: <https://docs.jetstream-cloud.org/getting-started/login/>
2. Select the allocation **CIS262025**.
   If it is not listed yet, make sure you have sent me your ACCESS username; it can take a day after I add you.

---

## 1. Create an SSH key (once, on your laptop)

Follow: <https://cvw.cac.cornell.edu/jetstream/keys/ssh-create>

In short, in a terminal (macOS/Linux Terminal, or Windows PowerShell):

```bash
ssh-keygen -t ed25519          # press Enter to accept the default location
cat ~/.ssh/id_ed25519.pub      # print your PUBLIC key
```

Copy the whole line that `cat` prints (it starts with `ssh-ed25519 ...`) and add it in Exosphere
(*SSH public keys* → upload/paste), or paste it while creating the instance in step 2.

> Only ever share the **`.pub`** file. The file without `.pub` is your private key; keep it secret.

---

## 2. Create your instance (VM)

Follow: <https://docs.jetstream-cloud.org/getting-started/first-instance/>
Video walkthrough: <https://cvw.cac.cornell.edu/jetstream/create-instance/configure-instance>

Settings:

| setting | value |
|---|---|
| Image | **Ubuntu 24.04** (featured image) |
| Name | **`ese670-<yourname>`** (e.g. `ese670-jsmith`) |
| Flavor | **`g3.large`** (half an A100 GPU, 20 GB GPU memory) |
| SSH key | the key from step 1 |

Click **Create** and wait until the instance status is **Ready**.
Find its **public IP address** (e.g. `149.165.xxx.xxx`) on the instance's page in Exosphere.

> Please use `g3.large` for assignments. Ask me before using the full-GPU `g3.xl` (e.g. for final projects).

---

## 3. Connect with SSH

On your laptop:

```bash
ssh exouser@149.165.xxx.xxx        # use YOUR instance's IP
```

- `exouser` is the user account that every Exosphere VM has (it is not your ACCESS username).
- The first time, answer `yes` to *"Are you sure you want to continue connecting?"*.

---

## 4. Set up the Python environment (once per VM)

Run these on the VM, inside the SSH session:

```bash
# 1. check the GPU (should list an NVIDIA A100)
nvidia-smi

# 2. create a Python environment (once)
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/ese670
source ~/ese670/bin/activate

# 3. install packages (CUDA wheels bundle their own CUDA runtime; only the driver is needed)
pip install --upgrade pip
pip install torch torchvision matplotlib numpy scipy tqdm jupyterlab

# 4. verify: should print the version, True, and the GPU name
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> Ubuntu 24.04 does not allow `pip install` into the system Python, which is why we use a
> virtual environment (`~/ese670`).

---

## 5. Start JupyterLab through an SSH tunnel

Open **another terminal on your laptop** and connect with a tunnel (`-L` forwards port 8888 from the VM to your laptop):

```bash
ssh -L 8888:localhost:8888 exouser@149.165.xxx.xxx
```

Then, on the VM in that same terminal:

```bash
source ~/ese670/bin/activate
jupyter lab --no-browser --port 8888
```

You will see a line like

```
http://localhost:8888/lab?token=cb9...
```

Copy it into your laptop's browser. Keep this terminal open while you work.

> If port 8888 is busy on your laptop, use `ssh -L 8889:localhost:8888 ...` and open `http://localhost:8889/...` instead.

---

## 6. Work

- **Upload** notebooks by dragging them into JupyterLab's file browser (left panel), or with the ⬆ button.
- **Download** results: right-click a file → *Download*.
- Files stay on the VM's disk (also while shelved). Download anything important; deleting the VM deletes its files.
- In a notebook, check the GPU with:
  ```python
  import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```

---

## 7. When you are done

### 7.1 Stop Jupyter
- Save your notebooks: **Ctrl+S**, or *File → Save*.
- Either use *File → Shut Down* in JupyterLab, or press **Ctrl+C** in the terminal where `jupyter lab` is running and answer `y`.

### 7.2 Close the connections on your laptop
- In the SSH terminal(s), type `exit` or press **Ctrl+D**.
- In a tunnel-only terminal (`ssh -N -L ...`), press **Ctrl+C**.

### 7.3 Shelve the VM: this is what stops the charges
Closing Jupyter and SSH does **not** stop billing: the VM keeps running and charges **32 SU/hour** for a `g3.large`.

- In Exosphere, open your instance → **Actions → Shelve**.
- The dialog asks whether to *release the public IP address*. You can **leave it unchecked**, so your VM keeps the same IP next time.
- Wait until the status shows **Shelved**.

| VM state | charge |
|---|---|
| Running | 100% |
| Suspended | 75% |
| Stopped | 50% |
| **Shelved** | **0%** |

**Shelve, don't stop.** Shelving keeps all your files and installed packages. Anything running is ended,
so let training finish (or save a checkpoint) first.

---

## Next time

1. Exosphere → your instance → **Actions → Unshelve**; wait for **Ready** (check the IP on the instance page).
2. Repeat **step 5**: open the tunnel, run `source ~/ese670/bin/activate`, start `jupyter lab`, and open the link.
   No need to repeat step 4.
3. When finished: **step 7** (shelve!).

---

## Notes

1. Long training stops when the laptop sleeps.  Run it inside `tmux` on the VM (`tmux new -s train`; detach with `Ctrl-b d`; reattach with `tmux attach -t train`). 

2. Attach a volume. A volume is a separate virtual disk that you attach to the VM. It comes from the project's 1 TB storage quota, uses no SUs, and lives on independently of any VM: you can detach it and attach it to another instance.

In Exosphere: Create → Volume. Give it a name (e.g. ese670-data) and a size, e.g. 200 GB.
On the volume's page: Attach it to your instance.
Exosphere normally mounts it automatically at `/media/volume/ese670-data`. Check on the VM:
```bash
   df -h | grep volume
   ls /media/volume/
```   
Use it for data, checkpoints and the virtual environment. A shortcut from your home folder helps:
```bash
   ln -s /media/volume/ese670-data ~/data
```

If you get "permission denied" when writing, run `sudo chown exouser:exouser /media/volume/ese670-data`.

Questions: Anton Selitskiy, anton.selitskiy@stonybrook.edu
