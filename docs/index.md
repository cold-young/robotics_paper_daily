---
layout: default
title: "HF Hot"
---

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

## Updated on 2026.09.22
> Usage instructions: [here]({{ site.baseurl }}/README.html#usage)

<style>.paper-nav{display:flex;flex-wrap:wrap;gap:.45rem;margin:1rem 0 1.25rem 0}.paper-nav a{border:1px solid #d0d7de;border-radius:999px;padding:.32rem .7rem;text-decoration:none;color:#24292f;background:#fff;font-size:.92rem}.paper-nav a.active{background:#0969da;color:#fff;border-color:#0969da}</style>

<nav class="paper-nav"><a href="{{ site.baseurl }}/" class="active">HF Hot</a>
<a href="{{ site.baseurl }}/conferences/corl-2025.html">CoRL 2025</a>
<a href="{{ site.baseurl }}/conferences/neurips-2025.html">NeurIPS 2025</a>
<a href="{{ site.baseurl }}/dexterous.html">Dexterous</a>
<a href="{{ site.baseurl }}/manipulation.html">Manipulation</a>
<a href="{{ site.baseurl }}/vla.html">VLA</a>
<a href="{{ site.baseurl }}/tactile.html">Tactile</a>
<a href="{{ site.baseurl }}/sim2real.html">Sim2Real</a></nav>

## 🔥 HuggingFace Hot Papers

| Publish Date | Title & Abstract | Authors | Links |
|:---------|:-----------------------|:---------|:------|
| **2026-09-21** | **CARE: Experience-Guided Atomic Corrective Execution for Vision-Language-Action Policies** `Manipulation` `VLA` 🔥 HF#12 💻 Code<br>Vision-Language-Action (VLA) policies achieve strong performance in robotic manipulation but remain brittle once execution deviates from nominal trajectories. We propose CARE (Corrective Atomic Robotic Execution), a framework that improves recovery by learning from failures encountered during execution. Instead of generating corrective data from manually designed or random perturbations, CARE... | Lijun Wang Team | [ArXiv](http://arxiv.org/abs/2609.24118) / [Code](https://github.com/xiaojunlan/care) / [HF](https://huggingface.co/papers/2609.24118) |
| **2026-09-20** | **Grounded Action Model: 3D Grounding as a Foundation for Robotics** `VLA` 🔥 HF#18 💻 Code<br>Manipulation policies must know which objects matter and where they are, yet the pretrained backbones that current robot foundation models build on, from language in vision-language-action models (VLAs) to video generation in world-action models (WAMs), do not directly require this metric grounding, leaving it to be learned implicitly from robot demonstrations. We propose Grounded Action Models... | Ranjay Krishna Team | [ArXiv](http://arxiv.org/abs/2609.23863) / [Code](https://github.com/GehaoZhang6/Grounded-Action-Model) / [HF](https://huggingface.co/papers/2609.23863) |
| **2026-09-21** | **Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies** `Dexterous` `VLA` 🔥 HF#22 💻 Code<br>Vision-Language-Action (VLA) models map observations to actions with no objective that accounts for how the world responds, so their robustness is bounded primarily by data coverage. World models carry precisely that missing objective and are better grounded for it, yet rolling the future forward costs seconds per decision and rules them out of the control loop. We show the two can be separated.... | Yong Jae Lee Team | [ArXiv](http://arxiv.org/abs/2609.24682) / [Code](https://github.com/trungdt880/THAW-VLA) / [HF](https://huggingface.co/papers/2609.24682) |
| **2026-09-09** | **HuRo: Robotizing Human Videos for Scalable VLA Pretraining** `Dexterous` `VLA` 🔥 HF#23<br>Human video datasets have emerged as a compelling alternative to expensive real-robot data, offering rich diversity at scale. To bridge the human-to-robot embodiment gap, existing approaches either robotize videos in task-matched settings or address observation and action alignment separately at scale. In this work, we systematically examine whether robotized human videos can provide effective... | Seon Joo Kim Team | [ArXiv](http://arxiv.org/abs/2609.10706) |
| **2026-09-19** | **Transferring the Intelligence of VLMs to Robotic Control** `VLA` 🔥 HF#26<br>Humans can seamlessly adapt to both physical and digital worlds, suggesting that while a digital-to-real gap exists in embodiment, environment and task, human intelligence itself may transfer across this gap. This naturally raises a fundamental question: can the intelligence of vision-language models (VLMs) similarly generalize from the digital world to the physical world for robotic control? We... | Shi-Min Hu Team | [ArXiv](http://arxiv.org/abs/2609.22966) / [Web](https://robodawn.top/) |
| **2026-09-18** | **From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention** `Dexterous` 🔥 HF#30<br>A pretrained robot foundation policy may execute most of a long-horizon task yet repeatedly fail at a few critical subtasks. Collecting additional full-task demonstrations for supervised fine-tuning (SFT) requires operators to repeat behaviors the policy already performs well. Reinforcement learning (RL) fine-tuning offers a promising path to bridge this gap, but existing approaches struggle to... | Lingfeng Sun Team | [ArXiv](http://arxiv.org/abs/2609.21788) / [Web](https://destiny000621.github.io/PARTS/) / [HF](https://huggingface.co/papers/2609.21788) |
| **2026-09-16** | **DeformSmith: Physics Harness-Guided Hierarchical Generation of Deformable Assets for Robot Manipulation** `Dexterous` `Manipulation` 🔥 HF#50<br>Creating deformable assets for robot manipulation requires jointly specifying their geometry, appearance, and physical properties. This is especially challenging for deformable objects, since text and images provide limited evidence about how they deform and respond to contact, yet these responses directly affect their suitability for interaction. Automated generation therefore needs to resolve... | Lei Sun Team | [ArXiv](http://arxiv.org/abs/2609.18620) / [Web](https://can-lee.github.io/deformsmith-web/) |

[contributors-shield]: https://img.shields.io/github/contributors/cold-young/robotics_paper_daily.svg?style=for-the-badge
[contributors-url]: https://github.com/cold-young/robotics_paper_daily/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/cold-young/robotics_paper_daily.svg?style=for-the-badge
[forks-url]: https://github.com/cold-young/robotics_paper_daily/network/members
[stars-shield]: https://img.shields.io/github/stars/cold-young/robotics_paper_daily.svg?style=for-the-badge
[stars-url]: https://github.com/cold-young/robotics_paper_daily/stargazers
[issues-shield]: https://img.shields.io/github/issues/cold-young/robotics_paper_daily.svg?style=for-the-badge
[issues-url]: https://github.com/cold-young/robotics_paper_daily/issues