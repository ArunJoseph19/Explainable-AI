# Contributing to Explainable AI

Thank you for your interest in contributing to the Explainable AI project! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, GPU specs)
- Relevant error messages or logs

### Suggesting Enhancements

We love new ideas! For feature requests:
- Check if the feature has already been requested
- Describe the feature and its benefits
- Provide examples of how it would work
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Explainable-AI.git
   cd Explainable-AI
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run existing notebooks to ensure nothing breaks
   # Test dashboards manually
   python dashboards/flux_generator_dashboard.py
   python dashboards/analysis_dashboard.py
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: Brief description of feature"
   # or
   git commit -m "Fix: Brief description of bug fix"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## 📝 Code Style Guidelines

### Python
- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

Example:
```python
def calculate_word_impact(prompt_dir: Path) -> Dict[str, float]:
    """
    Calculate impact scores from ablated images.
    
    Args:
        prompt_dir: Path to directory containing ablation results
        
    Returns:
        Dictionary mapping words to their impact scores
    """
    # Implementation here
    pass
```

### Documentation
- Update README.md if adding new features
- Add inline comments for complex logic
- Create or update notebooks for new workflows

### Commit Messages
Use conventional commit format:
- `Add:` New features
- `Fix:` Bug fixes
- `Update:` Changes to existing features
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Adding or updating tests

## 🧪 Testing

Before submitting:
1. Test all affected dashboards
2. Run relevant notebooks end-to-end
3. Check for any import errors
4. Verify GPU memory doesn't leak
5. Test with and without API keys configured

## 📋 Areas for Contribution

### High Priority
- [ ] Add unit tests for core functions
- [ ] Implement Docker deployment
- [ ] Add prompt template library
- [ ] Create REST API documentation
- [ ] Improve error handling and logging

### Medium Priority
- [ ] Support for additional image formats
- [ ] Batch processing capabilities
- [ ] Performance optimizations
- [ ] Additional LLM model support
- [ ] Internationalization (i18n)

### Documentation
- [ ] Video tutorials
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Performance benchmarks

## 🔍 Code Review Process

1. Maintainers will review your PR within 7 days
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged!

## 🌟 Recognition

Contributors will be:
- Listed in the project README
- Credited in release notes
- Invited to future project decisions

## 💬 Questions?

- Open a GitHub Discussion
- Comment on related issues
- Reach out to maintainers

Thank you for making Explainable AI better! 🎉
