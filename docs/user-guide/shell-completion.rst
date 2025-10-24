Shell Completion Setup
======================

Shell completion makes Homodyne's command-line interface faster and more discoverable by providing automatic suggestions as you type.

Why Use Shell Completion?
--------------------------

- **Faster typing:** Auto-complete long command names and options
- **Fewer errors:** Avoid typos in configuration file paths
- **Command discovery:** See available options without reading help text
- **Improved workflow:** Chain commands more efficiently

Bash Completion (Linux and macOS)
---------------------------------

**Verify Bash version:**

Bash completion requires Bash 4.0+ (macOS Catalina+ ships with Bash 3.2, upgrade if needed):

.. code-block:: bash

   bash --version

**Method 1: Using Runtime Script (Recommended)**

The homodyne package includes a completion script:

.. code-block:: bash

   source homodyne/runtime/shell/completion.sh

Add to ``~/.bashrc`` for persistent setup:

.. code-block:: bash

   echo 'source $(python -c "import homodyne; print(homodyne.__path__[0])")/runtime/shell/completion.sh' >> ~/.bashrc
   source ~/.bashrc

**Method 2: Manual Installation**

Generate and install completion:

.. code-block:: bash

   # Generate completion
   homodyne --generate-completion bash > ~/.homodyne-completion.bash

   # Add to ~/.bashrc
   echo "source ~/.homodyne-completion.bash" >> ~/.bashrc

   # Reload shell
   source ~/.bashrc

**Testing Bash Completion:**

.. code-block:: bash

   homodyne --<TAB>        # Shows available options
   homodyne --config <TAB> # Shows files in current directory
   homodyne --method <TAB> # Shows: nlsq, mcmc

Zsh Completion (Linux and macOS)
---------------------------------

**Verify Zsh installation:**

.. code-block:: bash

   zsh --version

**Method 1: Using Runtime Script (Recommended)**

.. code-block:: bash

   source homodyne/runtime/shell/completion.zsh

Add to ``~/.zshrc`` for persistent setup:

.. code-block:: bash

   echo 'source $(python -c "import homodyne; print(homodyne.__path__[0])")/runtime/shell/completion.zsh' >> ~/.zshrc
   exec zsh

**Method 2: Manual Installation**

Generate and install completion:

.. code-block:: bash

   # Generate completion
   homodyne --generate-completion zsh > ~/.homodyne-completion.zsh

   # Add to ~/.zshrc
   echo "source ~/.homodyne-completion.zsh" >> ~/.zshrc

   # Reload shell
   exec zsh

**Testing Zsh Completion:**

.. code-block:: bash

   homodyne --<TAB>        # Shows available options
   homodyne --config <TAB> # Shows files in current directory
   homodyne --method <TAB> # Shows: nlsq, mcmc

Fish Shell (Linux and macOS)
-----------------------------

Fish shell has built-in completion support. Homodyne completion scripts are compatible:

.. code-block:: bash

   homodyne --generate-completion fish | sudo tee /usr/share/fish/vendor_completions.d/homodyne.fish

macOS with Homebrew (Optional)
-------------------------------

If homodyne is installed via Homebrew, completion may be auto-configured:

.. code-block:: bash

   brew install bash-completion  # if needed
   # Completion typically available after installation

Troubleshooting Completion
--------------------------

**Completion not working:**

1. Verify homodyne is installed:

   .. code-block:: bash

      which homodyne
      homodyne --version

2. Verify completion script is sourced:

   .. code-block:: bash

      echo $BASH_COMPLETION_COMPAT_DIR   # Bash
      echo $fpath                        # Zsh

3. Reload shell configuration:

   .. code-block:: bash

      source ~/.bashrc   # Bash
      exec zsh           # Zsh

**No suggestions appearing:**

- For Bash: Ensure Bash 4.0+
- For Zsh: Check `~/.zshrc` includes completion source
- Verify completion script exists in correct location

**"Command not found" errors:**

Reinstall homodyne:

.. code-block:: bash

   pip install --upgrade homodyne
   source ~/.bashrc  # or exec zsh

Complete Example Setup
----------------------

**For Bash on Linux/macOS:**

.. code-block:: bash

   # Install homodyne (if not already installed)
   pip install homodyne

   # Generate and install completion
   homodyne --generate-completion bash > ~/.homodyne-completion.bash

   # Add to ~/.bashrc
   cat >> ~/.bashrc << 'EOF'
   # Homodyne bash completion
   if [ -f ~/.homodyne-completion.bash ]; then
     source ~/.homodyne-completion.bash
   fi
   EOF

   # Reload shell
   source ~/.bashrc

   # Test
   homodyne --<TAB>

**For Zsh on Linux/macOS:**

.. code-block:: bash

   # Install homodyne (if not already installed)
   pip install homodyne

   # Generate and install completion
   homodyne --generate-completion zsh > ~/.homodyne-completion.zsh

   # Add to ~/.zshrc
   cat >> ~/.zshrc << 'EOF'
   # Homodyne zsh completion
   if [ -f ~/.homodyne-completion.zsh ]; then
     source ~/.homodyne-completion.zsh
   fi
   EOF

   # Reload shell
   exec zsh

   # Test
   homodyne --<TAB>

Next Steps
----------

- :doc:`quickstart` - Run your first analysis
- :doc:`cli-usage` - CLI reference guide
- :doc:`examples` - Real-world workflow examples

See Also
--------

- :doc:`../developer-guide/performance` - Performance optimization
- PNAS paper: https://doi.org/10.1073/pnas.2401162121
