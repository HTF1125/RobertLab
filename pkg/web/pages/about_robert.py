import streamlit as st
from .base import BasePage


class AboutMe(BasePage):
    NAME = "Robert Han"
    DESCRIPTION = """
            Results-driven quantitative strategy and financial model developer
            with a strong background in buy-side equity investing.
            Offering 6 years of experience in financial research and strategy
            development, with a focus on developing solid and innovative
            investment solutions. Proficient in quantitative strategy
            development for institutional clients, with 2+ years of experience
            in delivering successful projects.
        """
    EMAIL = "hantianfeng@outlook.com"

    def render(self):
        st.title(self.NAME)
        self.low_margin_divider()

        st.write(self.DESCRIPTION)
        # st.download_button(
        #     label=" 📄 Download Resume",
        #     data=PDFbyte,
        #     file_name=resume_file.name,
        #     mime="application/octet-stream",
        # )
        st.write("📫", self.EMAIL)

        # --- EXPERIENCE & QUALIFICATIONS ---
        st.write("\n")
        st.subheader("Experience & Qulifications")
        self.low_margin_divider()

        st.write(
            """
            - ✔️ 6 Years expereince in quantitative strategy development.
            - ✔️ Strong hands on experience and knowledge in Python and Excel
            - ✔️ Good understanding of financial and statistical principles and their respective applications
            - ✔️ Excellent team-player and displaying strong sense of initiative on tasks
            """
        )

        # --- SKILLS ---
        st.write("\n")
        st.subheader("Hard Skills")
        self.low_margin_divider()

        st.write(
            """
            - 👩‍💻 Programming: Python, SQL, VBA
            - 📊 Data Visulization: PowerBi, MS Excel, Plotly
            - 📚 Modeling: Portoflio Optimizations, Factor & Risk Premia
            - 🗄️ Databases: Postgres, MySQL
            - 👩‍💻 Languages: Chinese, English, Korean, Japanese(Beginner)
            """
        )

        # --- WORK HISTORY ---
        st.write("\n")
        st.subheader("Work History")
        self.low_margin_divider()

        # --- JOB 1
        st.write("🚧", "**Quantitative Strategist | Dneuro Inc.**")
        st.write("05/2021 - 05/2023")
        st.write(
            """
            -► Spearheaded the development of a comprehensive Robo-Advisor project,
            encompassing all aspects of wealth management, involving goal-based
            dynamic asset allocation, market regime analysis, macro factors, and
            asset selection methodologies. Additionally, integrated behavioural
            finance to enhanced client experience.

            -► Led the US equity factor library construction project, taking
            responsibility for developing Python calculations for over 100
            fundamental factors. Designed and implemented a database operations
            module using MariaDB to facilitate efficient data management.

            -► Offered consulting services for OCIO Strategic Asset Allocation,
            utilizing simulations and portfolio optimization techniques. Developed
            a user-friendly web portal to facilitate client interactions and provide
            easy access to customized asset allocation strategies.
            """
        )

        self.low_margin_divider()
        # --- JOB 1
        st.write("🚧", "**Global Solutions Specialist | Woori Asset Management Corp.**")
        st.write("03/2017 - 05/2021")
        st.write(
            """
            -► Contributed to quantitative research and played a key role in
            developing models for optimizing global multi-asset portfolios.
            Leveraged advanced quantitative techniques to analyse market data,
            identify trends, and enhance portfolio performance and risk management.

            -► Contributed to global research, specifically focusing on US and
            Chinese equities. Built multiply financial analysis models in excel
            (VBA) with Bloomberg and Wind terminal.

            -► Managed liquidity positions for all global equity funds, ensuring
            efficient execution of Forex Hedges using futures and forward
            contracts. Proactively monitored market conditions and implemented
            hedging strategies to mitigate currency risks and optimize fund
            performance.
            """
        )
